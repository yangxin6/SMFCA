import torch
import torch.nn as nn

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)

@MODELS.register_module()
class DefaultSegmentorV3(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict)
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict)
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV4(nn.Module):
    def __init__(
            self,
            num_classes,
            backbone_out_channels,
            backbone=None,
            criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        seg_logits = self.seg_head(point.feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict)
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict)
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
            self,
            num_classes,
            backbone_out_channels,
            backbone=None,
            criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        seg_logits = self.seg_head(point.feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultMLossSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict:
            input_dict["condition"] = input_dict["condition"][0]

        seg_logits, aux_outputs = self.backbone(input_dict)

        if self.training:
            losses = {}
            main_loss = self.criteria(seg_logits, input_dict["segment"])
            losses["main_loss"] = main_loss

            total_loss = main_loss.clone()
            for i, aux_dict in enumerate(aux_outputs):
                pred_logit = aux_dict["pred_logit"]
                indices = aux_dict["indices"]

                # 直接使用 pred_logit 对应点的 label（需预对齐）
                target = input_dict["segment"][indices]  # 假设 batch index 0
                aux_loss = self.criteria(pred_logit, target) * 0.25
                losses[f"aux_loss_{i}"] = aux_loss
                total_loss += aux_loss

            losses["loss"] = total_loss
            return losses

        elif "segment" in input_dict:
            losses = {}
            main_loss = self.criteria(seg_logits, input_dict["segment"])
            losses["main_loss"] = main_loss

            total_loss = main_loss.clone()
            for i, aux_dict in enumerate(aux_outputs):
                pred_logit = aux_dict["pred_logit"]
                indices = aux_dict["indices"]

                # 直接使用 pred_logit 对应点的 label（需预对齐）
                target = input_dict["segment"][indices]  # 假设 batch index 0
                aux_loss = self.criteria(pred_logit, target) * 0.25
                losses[f"aux_loss_{i}"] = aux_loss
                total_loss += aux_loss

            losses["loss"] = total_loss
            losses["seg_logits"] = seg_logits
            return losses

        else:
            return dict(seg_logits=seg_logits)

from torch_scatter import scatter_mean, scatter_min, scatter_max
import torch.nn.functional as F

@MODELS.register_module()
class DefaultCRSegmentor(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def _cal_reg_loss(self, pred_off, grid, cluster, num_cls, label, eps=1e-4):
        # (a) 纯几何簇中心 μ_c  ── (P,3)
        geo_center = scatter_mean(grid, cluster, dim=0)

        # (b) 类别中心 μ_{c,l}
        flat_idx = cluster * num_cls + label  # [N]
        cls_center = scatter_mean(grid, flat_idx, dim=0,
                                  dim_size=geo_center.size(0) * num_cls)
        tgt_cls_center = cls_center[flat_idx]  # [N,3]

        # (c) 判断簇是否纯净
        lmin = scatter_min(label, cluster, dim=0)[0]
        lmax = scatter_max(label, cluster, dim=0)[0]
        pure = (lmin == lmax)  # (P)

        # (d) 最终目标中心
        tgt_center = torch.where(
            pure[cluster].unsqueeze(1),  # True→用 μ_c
            geo_center[cluster],  # [N,3]
            tgt_cls_center  # False→用 μ_{c,l}
        )

        tgt_offset = tgt_center - grid  # [N,3]

        # -------- 损失 --------
        #   · Smooth-L1 (长度)
        loss_L1 = F.smooth_l1_loss(pred_off, tgt_offset)

        #   · 方向余弦 (1-cos)
        pred_dir = F.normalize(pred_off, dim=1, eps=eps)
        tgt_dir = F.normalize(tgt_offset, dim=1, eps=eps)
        dir_cos = (pred_dir * tgt_dir).sum(1).clamp(-1, 1)
        loss_dir = (1.0 - dir_cos).mean()

        return loss_L1, loss_dir

    def cal_all_loss(self, input_dict, seg_logits, aux_outputs):
        seg_loss = self.criteria(seg_logits, input_dict["segment"])
        total_loss = seg_loss.clone()
        out_dict = {"seg_loss": seg_loss}

        grid = input_dict["grid_coord"].float()
        label = input_dict["segment"].long()  # [N]
        num_cls = int(seg_logits.shape[1])  # K
        grid_ = grid.clone()
        label_ = label.clone()
        for lv, aux in enumerate(aux_outputs):
            pred_off = aux["offset"]  # (M,3) or (N,3)
            cluster = aux["cluster"]  # (M)   or (N)
            fps_idx = aux.get("fps_idx")

            l1, dcos = self._cal_reg_loss(pred_off, grid_, cluster, num_cls, label_)

            out_dict[f"offset_L1_s{lv}"] = l1
            out_dict[f"offset_dir_s{lv}"] = dcos

            total_loss = total_loss + 0.25 * (l1 + dcos)

            grid_ = grid_[fps_idx]
            label_ = label_[fps_idx]
        out_dict["loss"] = total_loss
        return out_dict

    def forward(self, input_dict):
        seg_logits, aux_outputs = self.backbone(input_dict)   # aux_outputs: List[Dict]
        # train
        if self.training:
            loss = self.cal_all_loss(input_dict, seg_logits, aux_outputs)
            return loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.cal_all_loss(input_dict, seg_logits, aux_outputs)
            loss["seg_logits"] = seg_logits
            return loss
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV7(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict)
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict)
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


from torch_geometric.utils import scatter
from torch_geometric.nn.pool import voxel_grid
@MODELS.register_module()
class DefaultClusterSegmentor(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def _cal_reg_loss(self, pred_off, grid, big_size, num_cls, label, batch_, eps=1e-4):
        # 1. 大体素聚类（已按 batch 隔离）

        cluster = voxel_grid(grid, size=big_size, batch=batch_)
        # (a) 纯几何簇中心 μ_c  ── (P,3)
        geo_center = scatter_mean(grid, cluster, dim=0)
        # 2. (cluster, label) 局部类别中心
        flat_idx = cluster * num_cls + label  # [N]
        cls_center = scatter(grid, flat_idx, dim=0, reduce="mean")    # (⋯,3)
        tgt_cls_center = cls_center[flat_idx]  # [N,3]

        # 3. (batch, label) 全局类别中心
        key_glb   = batch_ * num_cls + label
        global_cls_center = scatter(grid, key_glb, dim=0, reduce="mean")  # (B·K,3)
        global_center = global_cls_center[key_glb]                     # (N,3)

        # 4. 判定簇纯净：同一 cluster 内 label 最大值 == 最小值
        lbl_min = scatter(label, cluster, dim=0, reduce="min")
        lbl_max = scatter(label, cluster, dim=0, reduce="max")
        pure_mask_pt = (lbl_min == lbl_max)[cluster]                  # (N,) bool
        
        # 5. 目标中心：纯簇 → tgt_cls_center；混簇 → global_center
        tgt_center = torch.where(
            pure_mask_pt.unsqueeze(-1), tgt_cls_center, global_center
        )
        tgt_offset = tgt_center - grid  # [N,3]

        # -------- 损失 --------
        #   · Smooth-L1 (长度)
        loss_L1 = F.smooth_l1_loss(pred_off, tgt_offset)

        #   · 方向余弦 (1-cos)
        pred_dir = F.normalize(pred_off, dim=1, eps=eps)
        tgt_dir = F.normalize(tgt_offset, dim=1, eps=eps)
        dir_cos = (pred_dir * tgt_dir).sum(1).clamp(-1, 1)
        loss_dir = (1.0 - dir_cos).mean()

        return loss_L1, loss_dir

    def cal_all_loss(self, input_dict, seg_logits, aux_outputs):
        seg_loss = self.criteria(seg_logits, input_dict["segment"])
        total_loss = seg_loss.clone()
        out_dict = {"seg_loss": seg_loss}

        grid = input_dict["grid_coord"].float()
        label = input_dict["segment"].long()  # [N]
        offset = input_dict["offset"] 
        batch = offset2batch(offset)
        num_cls = int(seg_logits.shape[1])  # K
        grid_ = grid.clone()
        label_ = label.clone()
        batch_ = batch.clone()
        for lv, aux in enumerate(aux_outputs):
            pred_off = aux["offset"]  # (M,3) or (N,3)
            big_size = aux["big_size"]  # (M)   or (N)
            down_idx = aux["down_idx"]  # (M)   or (N)

            l1, dcos = self._cal_reg_loss(pred_off, grid_, big_size, num_cls, label_, batch_)

            out_dict[f"offset_L1_s{lv}"] = l1
            out_dict[f"offset_dir_s{lv}"] = dcos

            total_loss = total_loss + 0.25 * (l1 + dcos)

            grid_ = grid_[down_idx]
            label_ = label_[down_idx]
            batch_ = batch_[down_idx]
        out_dict["loss"] = total_loss
        return out_dict

    def forward(self, input_dict):
        seg_logits, aux_outputs = self.backbone(input_dict)   # aux_outputs: List[Dict]
        # train
        if self.training:
            loss = self.cal_all_loss(input_dict, seg_logits, aux_outputs)
            return loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.cal_all_loss(input_dict, seg_logits, aux_outputs)
            loss["seg_logits"] = seg_logits
            return loss
        # test
        else:
            return dict(seg_logits=seg_logits)



import contextlib, time, torch

class GPUTimer:
    def __init__(self, dev):
        self.dev = dev
        if torch.cuda.is_available() and dev.type == "cuda":
            self.start_evt = torch.cuda.Event(enable_timing=True)
            self.end_evt   = torch.cuda.Event(enable_timing=True)
        else:  # CPU fallback
            self.start_evt = self.end_evt = None
            self.t0 = self.t1 = 0.0

    def __enter__(self):
        if self.start_evt is None:
            self.t0 = time.time()
        else:
            self.start_evt.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_evt is None:
            self.t1 = time.time()
            self.ms = (self.t1 - self.t0) * 1e3
        else:
            self.end_evt.record()
            self.end_evt.synchronize()
            self.ms = self.start_evt.elapsed_time(self.end_evt)

@contextlib.contextmanager
def timed(label: str, dev: torch.device, verbose: bool = True):
    timer = GPUTimer(dev)
    yield timer.__enter__()
    timer.__exit__(None, None, None)
    if verbose:
        print(f"{label:<20s}: {timer.ms:7.2f} ms")


# ---------- 64-bit FNV-1a 常量（已转成补码） ----------
FNV64_OFFSET = -0x340D631B7BDDDCDB  # 0xCBF29CE484222325 as signed int64
FNV64_PRIME  = 0x00000001000001B3   # 1099511628211  (< 2^63, 直接写)

def _pack_key(
    batch: torch.Tensor,  # (N,)
    cls:   torch.Tensor,  # (N,)
    vx:    torch.Tensor,  # (N,)
    vy:    torch.Tensor,  # (N,)
    vz:    torch.Tensor   # (N,)
) -> torch.Tensor:        # (N,) int64
    """
    Five-word FNV-1a 64-bit hash → signed int64.
    * 仅用 int64 运算；自然溢出即实现 2^64 回绕。
    * 在 CPU / CUDA 上均可运行。
    """
    data = torch.stack((batch, cls, vx, vy, vz), dim=1).long()

    h = torch.full((data.size(0),), FNV64_OFFSET,
                   dtype=torch.int64, device=data.device)

    for i in range(5):
        h ^= data[:, i]
        h *= FNV64_PRIME          # int64 会自动以 2^64 模回绕

    return h                      # (N,) int64  可直接 sort / searchsorted



@MODELS.register_module()
class DefaultOClusterSegmentor(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def _cal_reg_loss(
        self,
        pred_off:  torch.Tensor,       # (N,3)  预测 offset
        grid:      torch.Tensor,       # (N,3)  xyz
        base_grid: float,              # 体素尺寸
        num_cls:   int,                # 类别数
        label:     torch.Tensor,       # (N,)
        batch_id:  torch.Tensor,       # (N,)
        eps: float = 1e-4,
        verbose=False
    ):
        dev, N = grid.device, grid.size(0)
        grid_f = grid.float()

        # 1) coarse voxel
        with timed("voxel_grid", dev, verbose):
            cluster = voxel_grid(grid, size=base_grid, batch=batch_id)

        # 2) cluster stats
        with timed("cluster_stats", dev, verbose):
            cl_center = scatter(grid_f, cluster, dim=0, reduce="mean")          # (P,3)
            cl_batch  = scatter(batch_id, cluster, dim=0, reduce="min")
            cl_lbl_lo = scatter(label, cluster, dim=0, reduce="min")
            cl_lbl_hi = scatter(label, cluster, dim=0, reduce="max")
            pure_cl   = (cl_lbl_lo == cl_lbl_hi)
            pure_pt   = pure_cl[cluster]

        # 3) pure-cluster LUT
        with timed("pure-cluster", dev, verbose):
            cl_vox = scatter((grid / base_grid).floor().long(), cluster,
                            dim=0, reduce="min")
            pure_ix = pure_cl.nonzero(as_tuple=False).squeeze(1)
            pk = _pack_key(
                cl_batch[pure_ix], cl_lbl_lo[pure_ix],
                cl_vox[pure_ix, 0], cl_vox[pure_ix, 1], cl_vox[pure_ix, 2])
            if pk.numel():
                pk_sort, sidx = torch.sort(pk)
                pc_sort = cl_center[pure_ix][sidx]
            else:
                pk_sort = torch.empty(0, dtype=torch.int64, device=dev)
                pc_sort = torch.empty(0, 3, dtype=torch.float32, device=dev)

        # 4) global fallback
        key_bl   = batch_id * num_cls + label
        global_c = scatter(grid_f, key_bl, dim=0, reduce="mean")

        # ---------- 5)  ±K voxel search (K=3) ----------
        with timed("±1_voxel_search", dev, verbose):
            K = 1
            step_sign = (global_c[key_bl] - cl_center[cluster]).sign().long()
            tgt_c  = grid_f.clone()
            hit_pt = torch.zeros(N, dtype=torch.bool, device=dev)

            remain_mask = torch.ones(N, dtype=torch.bool, device=dev)
            for step in (1, K+1):
                if not remain_mask.any():
                    break 
                idx_remain = remain_mask.nonzero(as_tuple=False).squeeze(1)
                cand_vox = cl_vox[cluster[idx_remain]] + step_sign[idx_remain] * step
                cand_key = _pack_key(
                    batch_id[idx_remain], label[idx_remain],
                    cand_vox[:, 0], cand_vox[:, 1], cand_vox[:, 2]
                )

                idx = torch.searchsorted(pk_sort, cand_key)
                good = idx < pk_sort.numel()

                if good.any():                                         # 至少有候选
                    pk_good = pk_sort[idx[good]]                       # safe gather
                    hit_local = torch.zeros_like(good)
                    hit_local[good] = pk_good == cand_key[good]

                    if hit_local.any():
                        global_hit_idx = idx_remain[hit_local]
                        tgt_c[global_hit_idx] = pc_sort[idx[hit_local]]
                        hit_pt[global_hit_idx] = True
                        remain_mask[global_hit_idx] = False            # 已命中

        # # ③ 可视化
        # from project.utils import show_xyz_label
        # show_xyz_label(miss_xyz, miss_lbl)

        # ---------- 6) miss → 最近纯净簇   ——  ultra-simple ---------- #
        with timed("ultra-simple", dev, verbose):
            miss_pt = (~pure_pt) & (~hit_pt)
            miss_ix = miss_pt.nonzero(as_tuple=False).squeeze(1)  # (M,)
            if miss_ix.numel():

                # 1) 查询 miss
                pts_y = grid_f[miss_ix]  # (M,3)
                code_y = batch_id[miss_ix] * num_cls + label[miss_ix]

                # 2) 候选簇
                cand = cl_center[pure_ix]  # (C,3)
                code_x = cl_batch[pure_ix] * num_cls + cl_lbl_lo[pure_ix]

                # 3) 先按 code 排序，一次 group-by 循环
                uniq_code, inv_y = torch.unique(code_y, return_inverse=True)

                for g, code in enumerate(uniq_code):
                    # miss 组
                    m_mask = inv_y == g
                    m_idx = miss_ix[m_mask]  # (G,)
                    pts_g = pts_y[m_mask]  # (G,3)

                    # cand 组
                    c_mask = code_x == code
                    cand_g = cand[c_mask]  # (C,3)

                    MAX_ELEM = 256_0000          # 距离矩阵元素上限 ≈ 1 MB fp32
                    INF      = 1e38

                    if cand_g.numel():                                  # 有候选
                        G, C = pts_g.size(0), cand_g.size(0)

                        best_d = torch.full((G,), INF,  device=dev)     # per-miss 最小距离
                        best_i = torch.empty(G, dtype=torch.long, device=dev)

                        # ---- 动态决定沿哪一维切块 ----
                        if G * C <= MAX_ELEM:                           # 情况小 → 一次 cdist
                            dist   = torch.cdist(pts_g, cand_g)
                            nn_idx = dist.argmin(dim=1)
                            tgt_c[m_idx] = cand_g[nn_idx]
                            del dist
                            # torch.cuda.empty_cache()
                        else:                                           # 情况大 → 分块
                            if G >= C:                                  # miss 行数更大 → 切 G
                                STEP = max(MAX_ELEM // C, 1)
                                for g0 in range(0, G, STEP):
                                    g1 = min(g0 + STEP, G)
                                    dist = torch.cdist(pts_g[g0:g1], cand_g)   # (g,C)
                                    dmin, idx_local = dist.min(dim=1)
                                    mask = dmin < best_d[g0:g1]
                                    best_d[g0:g1][mask] = dmin[mask]
                                    best_i[g0:g1][mask] = idx_local[mask]
                                    del dist
                                    torch.cuda.empty_cache()
                            else:                                       # cand 列更多 → 切 C
                                STEP = max(MAX_ELEM // G, 1)
                                for c0 in range(0, C, STEP):
                                    c1 = min(c0 + STEP, C)
                                    dist = torch.cdist(pts_g, cand_g[c0:c1])   # (G,c)
                                    dmin, idx_local = dist.min(dim=1)
                                    mask = dmin < best_d
                                    best_d[mask] = dmin[mask]
                                    best_i[mask] = idx_local[mask] + c0
                                    del dist
                                    torch.cuda.empty_cache()

                            tgt_c[m_idx] = cand_g[best_i]


       
        # 7) loss
        with timed("loss_compute", dev, verbose):
            tgt_off = tgt_c - grid_f                       # (N,3)
            mag     = tgt_off.norm(dim=1)                  # (N,)
            # 0.99 分位阈值
            thresh  = torch.quantile(mag, 0.99)

            # L1 参与：所有 mag ≤ thresh
            mask_l1 = mag <= thresh                        # (N,)
            pred_l1 = pred_off[mask_l1]
            tgt_l1  = tgt_off[mask_l1]
            if mask_l1.any():
                loss_L1 = F.smooth_l1_loss(pred_l1, tgt_l1)
            else:
                loss_L1 = torch.tensor(0., device=dev)

            # Dir 参与：mag > 0 且 mag ≤ thresh
            mask_dir = (mag > 0) & (mag <= thresh)
            if mask_dir.any():
                pred_dir = pred_off[mask_dir]
                tgt_dir  = tgt_off[mask_dir]
                loss_dir = 1.0 - torch.cosine_similarity(
                    pred_dir, tgt_dir, dim=1, eps=eps
                ).mean()
            else:
                loss_dir = torch.tensor(0., device=dev)


            # loss_L1  = F.smooth_l1_loss(pred_off, tgt_off)
            # loss_dir = 1.0 - torch.cosine_similarity(pred_off, tgt_off, dim=1, eps=eps).mean()
        # with timed("loss_compute", dev, verbose):
        #     movable = ~pure_pt
        #     if movable.any():
        #         tgt_off = tgt_c[movable] - grid_f[movable]
        #         pred_m  = pred_off[movable]
        #         loss_L1  = F.smooth_l1_loss(pred_m, tgt_off)
        #         loss_dir = 1.0 - torch.cosine_similarity(
        #             pred_m, tgt_off, dim=1, eps=eps).mean()
        #     else:
        #         loss_L1 = loss_dir = torch.tensor(0., device=dev)

        # 8) log
        # hit_n  = hit_pt.sum().item()
        # miss_n = miss_pt.sum().item()
        # print(f"[hit]  {hit_n:6d} / {N}  ({hit_n/N:.1%})")
        # print(f"[miss] {miss_n:6d} / {N}  ({miss_n/N:.1%})")


        return loss_L1, loss_dir



    def cal_all_loss(self, input_dict, seg_logits, aux_outputs):
        seg_loss = self.criteria(seg_logits, input_dict["segment"])
        total_loss = seg_loss.clone()
        out_dict = {"seg_loss": seg_loss}

        grid = input_dict["grid_coord"].float()
        label = input_dict["segment"].long()  # [N]
        offset = input_dict["offset"] 
        batch = offset2batch(offset)
        num_cls = int(seg_logits.shape[1])  # K

        pred_off = aux_outputs["offset"] 
        base_gird = aux_outputs["base_gird"] 


        l1, dcos = self._cal_reg_loss(pred_off, grid, base_gird, num_cls, label, batch)

        out_dict[f"offset_L1"] = l1
        out_dict[f"offset_dir"] = dcos

        total_loss = total_loss + l1 + dcos 

        out_dict["loss"] = total_loss
        return out_dict

    def forward(self, input_dict):
        seg_logits, aux_outputs = self.backbone(input_dict)   # aux_outputs: List[Dict]
        # train
        if self.training:
            loss = self.cal_all_loss(input_dict, seg_logits, aux_outputs)
            return loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.cal_all_loss(input_dict, seg_logits, aux_outputs)
            loss["seg_logits"] = seg_logits
            return loss
        # test
        else:
            return dict(seg_logits=seg_logits)



@MODELS.register_module()
class DefaultSemRegSegmentor(nn.Module):
    def __init__(
            self,
            backbone=None,
            criteria=None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def cal_all_loss(self, input_dict, seg_logits, pred_v):
        # 1) 分割 loss
        seg_loss = self.criteria(seg_logits, input_dict["segment"])
        total_loss = seg_loss.clone()
        out_dict = {"seg_loss": seg_loss}

        # 2) 回归 shift 向量 loss
        grid      = input_dict["grid_coord"].float()      # (N,3)，可选不用
        target_v  = input_dict["shift_v"].float()        # (N,3)

        # 只对那些有“位移”的点计算
        mag = target_v.norm(dim=1)                       # (N,)
        mask = mag > 0

        if mask.any():
            # L1 长度 loss
            l1 = F.smooth_l1_loss(pred_v[mask], target_v[mask])
            # 方向 loss：cosine similarity
            cos_sim = F.cosine_similarity(pred_v[mask], target_v[mask], dim=1, eps=1e-6)
            dcos = (1.0 - cos_sim).mean()
        else:
            # 若没有任何点需要移动，则置 0
            l1 = torch.tensor(0., device=seg_logits.device)
            dcos = torch.tensor(0., device=seg_logits.device)

        l1 *= 0.1
        out_dict["offset_L1"] = l1
        out_dict["offset_dir"] = dcos

        # 3) 汇总
        total_loss = total_loss + l1 + dcos
        out_dict["loss"] = total_loss
        return out_dict

    def forward(self, input_dict):
        seg_logits, pred_v = self.backbone(input_dict)
        # 训练阶段
        if self.training:
            return self.cal_all_loss(input_dict, seg_logits, pred_v)
        # 验证阶段：也计算 loss 并带出 seg_logits
        elif "segment" in input_dict:
            loss_dict = self.cal_all_loss(input_dict, seg_logits, pred_v)
            loss_dict["seg_logits"] = seg_logits
            return loss_dict
        # 测试/推理
        else:
            return {"seg_logits": seg_logits}


@MODELS.register_module()
class DefaultInstanceSegmentor(nn.Module):
    def __init__(
            self,
            backbone=None,
            criteria=None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict)
            return loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict)
            loss.update(seg_logits)
            return loss
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
            self,
            backbone=None,
            criteria=None,
            num_classes=40,
            backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        feat = self.backbone(input_dict)
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
