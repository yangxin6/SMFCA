from functools import partial
import torch
import torch.nn as nn
from einops import rearrange
import spconv.pytorch as spconv
from timm.models.layers import trunc_normal_

from ..builder import MODELS, build_model
from ..utils import offset2batch
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.utils import scatter
from torch_cluster import radius

# from project.organ_seg.data_process.data_show.show import show_pcd2, show_pcd0


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_fn=None,
        indice_key=None,
        depth=4,
        groups=None,
        grid_size=None,
        bias=False,
    ):
        super().__init__()
        assert embed_channels % groups == 0
        self.groups = groups
        self.embed_channels = embed_channels
        self.proj = nn.ModuleList()
        self.grid_size = grid_size
        self.weight = nn.ModuleList()
        self.l_w = nn.ModuleList()
        self.proj.append(
            nn.Sequential(
                nn.Linear(embed_channels, embed_channels, bias=False),
                norm_fn(embed_channels),
                nn.ReLU(),
            )
        )
        for _ in range(depth - 1):
            self.proj.append(
                nn.Sequential(
                    nn.Linear(embed_channels, embed_channels, bias=False),
                    norm_fn(embed_channels),
                    nn.ReLU(),
                )
            )
            self.l_w.append(
                nn.Sequential(
                    nn.Linear(embed_channels, embed_channels, bias=False),
                    norm_fn(embed_channels),
                    nn.ReLU(),
                )
            )
            self.weight.append(nn.Linear(embed_channels, embed_channels, bias=False))

        self.adaptive = nn.Linear(embed_channels, depth - 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Linear(embed_channels * 2, embed_channels, bias=False),
            norm_fn(embed_channels),
            nn.ReLU(),
        )
        self.voxel_block = spconv.SparseSequential(
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels),
        )
        self.act = nn.ReLU()

    def forward(self, x, clusters):
        feat = x.features
        feats = []
        for i, cluster in enumerate(clusters):
            pw = self.l_w[i](feat)
            pw = pw - scatter(pw, cluster, reduce="mean")[cluster]
            pw = self.weight[i](pw)
            pw = torch.exp(pw - pw.max())
            pw = pw / (scatter(pw, cluster, reduce="sum", dim=0)[cluster] + 1e-6)
            pfeat = self.proj[i](feat) * pw
            pfeat = scatter(pfeat, cluster, reduce="sum")[cluster]
            feats.append(pfeat)
        adp = self.adaptive(feat)
        adp = torch.softmax(adp, dim=1)
        feats = torch.stack(feats, dim=1)
        feats = torch.einsum("l n, l n c -> l c", adp, feats)
        feat = self.proj[-1](feat)
        feat = torch.cat([feat, feats], dim=1)
        feat = self.fuse(feat) + x.features
        res = feat
        x = x.replace_feature(feat)
        x = self.voxel_block(x)
        x = x.replace_feature(self.act(x.features + res))
        return x


class DonwBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        depth,
        sp_indice_key,
        point_grid_size,
        num_ref=16,
        groups=None,
        norm_fn=None,
        sub_indice_key=None,
    ):
        super().__init__()
        self.num_ref = num_ref
        self.depth = depth
        self.point_grid_size = point_grid_size
        self.down = spconv.SparseSequential(
            spconv.SparseConv3d(
                in_channels,
                embed_channels,
                kernel_size=2,
                stride=2,
                indice_key=sp_indice_key,
                bias=False,
                padding=1,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                BasicBlock(
                    in_channels=embed_channels,
                    embed_channels=embed_channels,
                    depth=len(point_grid_size) + 1,
                    groups=groups,
                    grid_size=point_grid_size,
                    norm_fn=norm_fn,
                    indice_key=sub_indice_key,
                )
            )

    def forward(self, x):
        x = self.down(x)
        coord = x.indices[:, 1:].float()
        batch = x.indices[:, 0]
        clusters = []
        for grid_size in self.point_grid_size:
            cluster = voxel_grid(pos=coord, size=grid_size, batch=batch)
            _, cluster = torch.unique(cluster, return_inverse=True)
            clusters.append(cluster)
        for block in self.blocks:
            x = block(x, clusters)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        embed_channels,
        depth,
        sp_indice_key,
        norm_fn=None,
        down_ratio=2,
        sub_indice_key=None,
    ):
        super().__init__()
        assert depth > 0
        self.up = spconv.SparseSequential(
            spconv.SparseInverseConv3d(
                in_channels,
                embed_channels,
                kernel_size=down_ratio,
                indice_key=sp_indice_key,
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList()
        self.fuse = nn.Sequential(
            nn.Linear(skip_channels + embed_channels, embed_channels),
            norm_fn(embed_channels),
            nn.ReLU(),
            nn.Linear(embed_channels, embed_channels),
            norm_fn(embed_channels),
            nn.ReLU(),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = x.replace_feature(
            self.fuse(torch.cat([x.features, skip_x.features], dim=1)) + x.features
        )
        return x


@MODELS.register_module("OACNNs-v3m1")
class OACNNs(nn.Module):
    def __init__(
        self,
        backbone,
        in_channels,
        num_classes,
        embed_channels=64,
        enc_num_ref=[16, 16, 16, 16],
        enc_channels=[64, 64, 128, 256],
        groups=[2, 4, 8, 16],
        enc_depth=[2, 3, 6, 4],
        down_ratio=[2, 2, 2, 2],
        dec_channels=[96, 96, 128, 256],
        point_grid_size=[[16, 32, 64], [8, 16, 24], [4, 8, 12], [2, 4, 6]],
        dec_depth=[2, 2, 2, 2],
        second_epoch=100,
        two_ignore_label=[3],
        mask_scale=1.5
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_channels)
        self.embed_channels = embed_channels
        self.second_epoch = second_epoch
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.backbone = build_model(backbone)
        self.seg_head = nn.Linear(embed_channels, num_classes)
        self.two_ignore_label = two_ignore_label
        self.mask_scale = mask_scale

        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
        )

        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(self.num_stages):
            self.enc.append(
                DonwBlock(
                    in_channels=embed_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=enc_channels[i],
                    depth=enc_depth[i],
                    norm_fn=norm_fn,
                    groups=groups[i],
                    point_grid_size=point_grid_size[i],
                    num_ref=enc_num_ref[i],
                    sp_indice_key=f"spconv{i}",
                    sub_indice_key=f"subm{i + 1}",
                )
            )
            self.dec.append(
                UpBlock(
                    in_channels=(
                        enc_channels[-1]
                        if i == self.num_stages - 1
                        else dec_channels[i + 1]
                    ),
                    skip_channels=embed_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=dec_channels[i],
                    depth=dec_depth[i],
                    norm_fn=norm_fn,
                    sp_indice_key=f"spconv{i}",
                    sub_indice_key=f"subm{i}",
                )
            )

        self.final = spconv.SubMConv3d(dec_channels[0], num_classes, kernel_size=1)

        self.loss_full = nn.CrossEntropyLoss()
        self.loss_crop = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

    def expand_mask(self, coords, mask, radius_value, batch):
        """
        使用 torch_cluster.radius 扩展掩码区域。

        Args:
            coords (torch.Tensor): 点云坐标 (N, 3)
            mask (torch.Tensor): 初始布尔掩码 (N,)
            radius_value (float): 半径范围，单位为米
            batch (torch.Tensor, optional): 每个点的 batch 索引 (N,)

        Returns:
            torch.Tensor: 扩展后的布尔掩码 (N,)
        """
        coords = coords.float()

        # 提取被 mask 标记的点
        target_coords = coords[mask]  # 目标点坐标 (M, 3)
        target_batch = batch[mask]  # 目标点对应的 batch 索引

        # 使用 torch_cluster.radius 查找在 radius 范围内的点
        idx_pairs = radius(coords, target_coords, radius_value, batch_x=batch, batch_y=target_batch)

        # 生成全局掩码
        expanded_mask = torch.zeros(coords.size(0), dtype=torch.bool, device=coords.device)
        expanded_mask[idx_pairs[1]] = True

        return expanded_mask

    def pred_feat(self, input_dict):
        backbone_feat = self.backbone(input_dict)
        seg_pred = self.seg_head(backbone_feat)
        # if input_dict['epoch'] < self.second_epoch:
        #     return seg_pred, None, None

        pred_labels = torch.argmax(seg_pred, dim=1)  # pred_labels shape: (N,)
        mask = ~torch.isin(pred_labels, torch.tensor(self.two_ignore_label, device=pred_labels.device))

        offset = input_dict["offset"]
        o_batch = offset2batch(offset)

        expanded_mask = self.expand_mask(input_dict["grid_coord"], mask, radius_value=0.05 / 0.02, batch=o_batch)

        discrete_coord = input_dict["grid_coord"][expanded_mask]
        # segment = input_dict["segment"][expanded_mask]
        feat = input_dict["feat"][expanded_mask]
        # feat2 = backbone_feat[expanded_mask]
        batch = o_batch[expanded_mask]

        # from project.organ_seg.data_process.data_show.show import show_pcd2, show_pcd0
        # show_pcd2(input_dict["grid_coord"][expanded_mask].cpu().numpy(), input_dict["segment"][expanded_mask].cpu().numpy())

        # feat = torch.cat([feat1, feat2], dim=1)

        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat([batch.unsqueeze(-1), discrete_coord], dim=1)
            .int()
            .contiguous(),
            spatial_shape=torch.add(
                torch.max(discrete_coord, dim=0).values, 1
            ).tolist(),
            batch_size=batch[-1].tolist() + 1,
        )

        x = self.stem(x)
        skips = [x]
        for i in range(self.num_stages):
            x = self.enc[i](x)
            skips.append(x)
        x = skips.pop(-1)
        for i in reversed(range(self.num_stages)):
            skip = skips.pop(-1)
            x = self.dec[i](x, skip)
        x = self.final(x).features
        crop_feat = x
        full_features = torch.zeros(
            (input_dict["grid_coord"].shape[0], self.num_classes),  # 原始点数量 x 特征维度
            device=x.device,
            dtype=x.dtype,
        )
        full_features[expanded_mask] = x
        full_features[~expanded_mask] = seg_pred[~expanded_mask]

        return full_features, crop_feat, expanded_mask

    def expand_labels(self, input_dict, radius_value=0.5):
        """
        对每个类别单独扩展标签区域。

        Args:
            radius_value (float): 扩展的半径范围
        Returns:
            torch.Tensor: 扩展后的标签 (N,)
        """
        segment = input_dict["segment"]
        # expand segment
        grid_coords = input_dict["grid_coord"].float()
        offset = input_dict["offset"]
        batch = offset2batch(offset)

        # 初始化扩展后的标签
        expanded_labels = segment.clone()

        # 转为 Tensor 格式的 ignore_label
        ignore_label_tensor = torch.tensor(self.two_ignore_label, device=segment.device)

        # 获取需要扩展的类别（所有非 ignore_label 的类别）
        target_classes = torch.unique(segment[~torch.isin(segment, ignore_label_tensor)])

        # 分别对每个类别进行扩展
        for cls in target_classes:
            # 找到属于当前类别的点
            cls_mask = (segment == cls)
            target_coords = grid_coords[cls_mask]
            target_batch = batch[cls_mask]

            # 使用 torch_cluster.radius 扩展区域
            idx_pairs = radius(
                grid_coords,  # 所有点的坐标
                target_coords,  # 当前类别的目标点
                radius_value,  # 半径范围
                batch_x=batch,  # 所有点的批次索引
                batch_y=target_batch,  # 当前类别点的批次索引
            )

            # 将扩展区域的点标签设置为当前类别
            expanded_labels[idx_pairs[1]] = cls

        return expanded_labels

    def loss(self, input_dict, full_feat, crop_feat, expanded_mask):
        # expand segment
        expand_segment = self.expand_labels(input_dict, 0.05 / 0.02)

        if crop_feat is None:
            full_loss = self.loss_full(full_feat, expand_segment)
            return_dict = dict(
                loss=full_loss,
            )
            return return_dict

        segment = input_dict["segment"]
        crop_segment = segment[expanded_mask]

        full_loss = self.loss_full(full_feat, segment)
        crop_loss = self.loss_crop(crop_feat, crop_segment)

        loss = 0.2 * full_loss + 0.8 * crop_loss
        return_dict = dict(
            loss=loss,
            full_loss=full_loss,
            crop_loss=crop_loss,
        )
        return return_dict

    def forward(self, input_dict):
        full_feat, crop_feat, expanded_mask = self.pred_feat(input_dict)
        # train
        if self.training:
            return_dict = self.loss(input_dict, full_feat, crop_feat, expanded_mask)
        # eval
        elif "segment" in input_dict.keys():
            return_dict = self.loss(input_dict, full_feat, crop_feat, expanded_mask)
            return_dict['seg_logits']=full_feat
        # test
        else:
            return_dict = dict(seg_logits=full_feat)

        return return_dict

    # def forward(self, input_dict):
    #     discrete_coord = input_dict["grid_coord"]
    #     feat = input_dict["feat"]
    #     offset = input_dict["offset"]
    #     batch = offset2batch(offset)
    #     x = spconv.SparseConvTensor(
    #         features=feat,
    #         indices=torch.cat([batch.unsqueeze(-1), discrete_coord], dim=1)
    #         .int()
    #         .contiguous(),
    #         spatial_shape=torch.add(
    #             torch.max(discrete_coord, dim=0).values, 1
    #         ).tolist(),
    #         batch_size=batch[-1].tolist() + 1,
    #     )
    #
    #     x = self.stem(x)
    #     skips = [x]
    #     for i in range(self.num_stages):
    #         x = self.enc[i](x)
    #         skips.append(x)
    #     x = skips.pop(-1)
    #     for i in reversed(range(self.num_stages)):
    #         skip = skips.pop(-1)
    #         x = self.dec[i](x, skip)
    #     x = self.final(x)
    #     return x.features

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
