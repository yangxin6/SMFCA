"""
SparseUNet Driven by SpConv (recommend)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import math
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv.pytorch as spconv
from torch_geometric.utils import scatter

from timm.models.layers import trunc_normal_

from pointcept.models import batch2offset
from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch
from torch_scatter import scatter_mean, scatter_max


# ---------- 辅助函数 ----------
def sc_get_batch_indices(x: spconv.SparseConvTensor):
    """返回每个 voxel 所属 batch idx (N,)."""
    return x.indices[:, 0]

def sc_replace_features(x: spconv.SparseConvTensor,
                        new_feat: torch.Tensor) -> spconv.SparseConvTensor:
    """用 new_feat 替换 features，保持坐标不变。"""
    return spconv.SparseConvTensor(
        features=new_feat,
        indices=x.indices,
        spatial_shape=x.spatial_shape,
        batch_size=x.batch_size
    )

class MultiFourierDecoupling(nn.Module):
    """返回三路 (N,C) 特征，dtype 与输入完全一致"""
    def __init__(self, cutoff=0.25):
        super().__init__()
        self.cutoff = cutoff

    @staticmethod
    @torch.no_grad()
    def _make_k_grid(D, H, W, device):
        key = (D, H, W, device)
        if not hasattr(MultiFourierDecoupling, "_cache"): MultiFourierDecoupling._cache = {}
        if key not in MultiFourierDecoupling._cache:
            kx = torch.fft.fftfreq(D, device=device)
            ky = torch.fft.fftfreq(H, device=device)
            kz = torch.fft.rfftfreq(W, device=device)
            MultiFourierDecoupling._cache[key] = torch.sqrt(
                torch.meshgrid(kx, ky, kz, indexing='ij')[0]**2 +
                torch.meshgrid(kx, ky, kz, indexing='ij')[1]**2 +
                torch.meshgrid(kx, ky, kz, indexing='ij')[2]**2
            )
        return MultiFourierDecoupling._cache[key]

    def forward(self, st: spconv.SparseConvTensor):
        dtype_in = st.features.dtype
        device   = st.features.device
        B        = st.batch_size
        D, H, W  = st.spatial_shape
        b, z, y, x = st.indices.T

        dense = st.dense().float()                          # fp32 FFT
        X = torch.fft.rfftn(dense, dim=(-3, -2, -1))
        del dense

        k_norm = self._make_k_grid(D, H, W, device)
        masks = [
            (k_norm <= self.cutoff),                        # low
            ~( (k_norm <= self.cutoff) | (k_norm >= 1-self.cutoff) ),  # mid
            (k_norm >= 1 - self.cutoff)                     # high
        ]

        out_feats = []
        for m in masks:
            part  = torch.fft.irfftn(X * m, s=(D, H, W), dim=(-3, -2, -1))
            feats = part[b, :, z, y, x].to(dtype_in).contiguous()
            out_feats.append(feats)
            del part, feats
        del X; torch.cuda.empty_cache()

        return tuple(out_feats)   # (low_feats, mid_feats, high_feats)

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttentionFusion, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q_feature: torch.Tensor,
                K_feature: torch.Tensor,
                offset: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        Q_feature : (N_total, C)
        K_feature : (N_total, C)
        offset    : (B,)  累计长度，例如 [n0, n0+n1, n0+n1+n2, ...]
        """
        N_total, C = Q_feature.shape  # 所有点拼在一起的长度与通道数
        Q = self.query(Q_feature)  # (N_total, C)
        K = self.key(K_feature)  # (N_total, C)
        V = self.value(K_feature)  # (N_total, C)

        start = 0
        out_chunks = []
        for end in offset:  # 按 batch 切片
            q_i = Q[start:end]  # (n_i, C)
            k_i = K[start:end]  # (n_i, C)
            v_i = V[start:end]  # (n_i, C)

            # scaled dot‑product attention
            scores_i = (q_i @ k_i.T) / math.sqrt(C)  # (n_i, n_i)
            attn_i = F.softmax(scores_i, dim=-1)  # (n_i, n_i)
            out_i = attn_i @ v_i  # (n_i, C)

            out_chunks.append(out_i)
            start = end

        attended_features = torch.cat(out_chunks, dim=0)  # (N_total, C)
        return attended_features

class SMFAFusionBlock(nn.Module):
    def __init__(self, d_model, cutoff=0.30):
        super().__init__()
        self.fourier_transform = MultiFourierDecoupling(cutoff)
        self.cross_low  = CrossAttentionFusion(embed_dim=d_model)
        self.cross_high = CrossAttentionFusion(embed_dim=d_model)

        # 三个可学习标量 → softmax 后自动归一化
        self.alpha = nn.Parameter(torch.zeros(3))   # 初始全 0 ⇒ 均分 1/3
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x0: spconv.SparseConvTensor):
        x_low, x_mid, x_high = self.fourier_transform(x0)
        offset = batch2offset(x0.indices[:, 0])

        low_new  = self.cross_low (x_mid, x_low,  offset)
        high_new = self.cross_high(x_mid, x_high, offset)

        # ---- 可学习加权融合 ----
        w = torch.softmax(self.alpha, dim=0)        # (3,) 且 ∑w=1
        band_fused = w[0]*low_new + w[1]*x_mid + w[2]*high_new

        fused = self.act(x0.features + band_fused)
        return sc_replace_features(x0, fused.contiguous())


class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
            self,
            in_channels,
            embed_channels,
            stride=1,
            norm_fn=None,
            indice_key=None,
            bias=False,
    ):
        super().__init__()

        assert norm_fn is not None

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, embed_channels, kernel_size=1, bias=False
                ),
                norm_fn(embed_channels),
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out


@MODELS.register_module("SMFCA-v4m0")
class SMFANetBase(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            base_channels=32,
            channels=(32, 64, 128, 256, 256, 128, 96, 96),
            layers=(2, 3, 4, 6, 2, 2, 2, 2),
            cls_mode=False,
            return_decoder_feat=False,
    ):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        self.cls_mode = cls_mode
        self.return_decoder_feat = return_decoder_feat

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(base_channels),
            nn.ReLU(),
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList() if not self.cls_mode else None

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(
                        enc_channels,
                        channels[s],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                    ),
                    norm_fn(channels[s]),
                    nn.ReLU(),
                )
            )
            self.enc.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            # (f"block{i}", block(enc_channels, channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"))
                            # if i == 0 else
                            (
                                f"block{i}",
                                block(
                                    channels[s],
                                    channels[s],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s + 1}",
                                ),
                            )
                            for i in range(layers[s])
                        ]
                    )
                )
            )
            if not self.cls_mode:
                # decode num_stages
                self.up.append(
                    spconv.SparseSequential(
                        spconv.SparseInverseConv3d(
                            channels[len(channels) - s - 2],
                            dec_channels,
                            kernel_size=2,
                            bias=False,
                            indice_key=f"spconv{s + 1}",
                        ),
                        norm_fn(dec_channels),
                        nn.ReLU(),
                    )
                )
                self.dec.append(
                    spconv.SparseSequential(
                        OrderedDict(
                            [
                                (
                                    (
                                        f"block{i}",
                                        block(
                                            dec_channels + enc_channels,
                                            dec_channels,
                                            norm_fn=norm_fn,
                                            indice_key=f"subm{s}",
                                        ),
                                    )
                                    if i == 0
                                    else (
                                        f"block{i}",
                                        block(
                                            dec_channels,
                                            dec_channels,
                                            norm_fn=norm_fn,
                                            indice_key=f"subm{s}",
                                        ),
                                    )
                                )
                                for i in range(layers[len(channels) - s - 1])
                            ]
                        )
                    )
                )

            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        # self.multi_scale_fusion = MultiScaleAttnFusion(in_channels=channels[:3])
        self.fourier_attn = SMFAFusionBlock(d_model=channels[3])
        final_in_channels = (
            channels[-1] if not self.cls_mode else channels[self.num_stages - 1]
        )
        self.final = (
            spconv.SubMConv3d(
                final_in_channels, num_classes, kernel_size=1, padding=1, bias=True
            )
            if num_classes > 0
            else spconv.Identity()
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_dict):
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]

        batch = offset2batch(offset)
        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat(
                [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1,
        )
        x = self.conv_input(x)
        skips = [x]

        # enc forward
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)

        # skip_fused = self.multi_scale_fusion(skips[1:4])
        # skips[1:4] = skip_fused
        x = skips.pop(-1)

        x_f = self.fourier_attn(x)
        x = x.replace_feature(x_f.features)

        if self.return_decoder_feat:
            decoder_feats = [x]
        if not self.cls_mode:
            # dec forward
            for s in reversed(range(self.num_stages)):
                x = self.up[s](x)
                skip = skips.pop(-1)
                x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
                x = self.dec[s](x)
                if self.return_decoder_feat:
                    decoder_feats.append(x)

        x = self.final(x)
        if self.cls_mode:
            x = x.replace_feature(
                scatter(x.features, x.indices[:, 0].long(), reduce="mean", dim=0)
            )
        if self.return_decoder_feat:
            return decoder_feats
        return x.features

