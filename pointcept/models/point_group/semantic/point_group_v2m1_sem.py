"""
PointGroup for instance segmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Chengyao Wang
Please cite our work if the code is helpful to you.
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointgroup_ops import ballquery_batch_p, bfs_cluster

from pointcept.models.utils import offset2batch, batch2offset

from pointcept.models.builder import MODELS, build_model



@MODELS.register_module("PG-v2m1-sm")
class PointGroup(nn.Module):
    def __init__(
            self,
            backbone,
            backbone_out_channels=64,
            semantic_num_classes=20,
            semantic_ignore_index=-1,
            segment_ignore_index=(-1, 0, 1),
            instance_ignore_index=-1,
            cluster_thresh=1.5,
            cluster_closed_points=300,
            cluster_propose_points=100,
            cluster_min_points=50,
            voxel_size=0.02,
            class_weights=None,
            test=False,
    ):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.semantic_num_classes = semantic_num_classes
        self.segment_ignore_index = segment_ignore_index
        self.semantic_ignore_index = semantic_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.cluster_thresh = cluster_thresh
        self.cluster_closed_points = cluster_closed_points
        self.cluster_propose_points = cluster_propose_points
        self.cluster_min_points = cluster_min_points
        self.voxel_size = voxel_size
        self.test = test
        self.backbone = build_model(backbone)
        self.bias_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            norm_fn(backbone_out_channels),
            nn.ReLU(),
            nn.Linear(backbone_out_channels, 3),
        )

        self.seg_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            norm_fn(backbone_out_channels),
            nn.ReLU(),
            nn.Linear(backbone_out_channels, semantic_num_classes),
        )

        self.class_weights = torch.tensor(class_weights) if class_weights is not None else torch.ones(
            semantic_num_classes)
        self.ce_criteria = torch.nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=semantic_ignore_index)


    def loss(self, logit_pred, bias_pred, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        instance_centroid = data_dict["instance_centroid"]
        seg_loss = self.ce_criteria(logit_pred, segment)
        mask = (instance != self.instance_ignore_index).float()

        # mask = ~torch.isin(instance, torch.tensor(self.segment_ignore_index).to(instance.device))
        # mask = mask.to(torch.float)

        bias_gt = instance_centroid - coord
        bias_dist = torch.sum(torch.abs(bias_pred - bias_gt), dim=-1)
        bias_l1_loss = torch.sum(bias_dist * mask) / (torch.sum(mask) + 1e-8)

        bias_pred_norm = bias_pred / (
                torch.norm(bias_pred, p=2, dim=1, keepdim=True) + 1e-8
        )
        bias_gt_norm = bias_gt / (torch.norm(bias_gt, p=2, dim=1, keepdim=True) + 1e-8)
        cosine_similarity = -(bias_pred_norm * bias_gt_norm).sum(-1)
        bias_cosine_loss = torch.sum(cosine_similarity * mask) / (
                torch.sum(mask) + 1e-8
        )

        loss = seg_loss + bias_l1_loss + bias_cosine_loss
        return_dict = dict(
            loss=loss,
            seg_loss=seg_loss,
            bias_l1_loss=bias_l1_loss,
            bias_cosine_loss=bias_cosine_loss,
        )
        return return_dict

    def pred_feat(self, logit_pred, bias_pred, data_dict):
        coord = data_dict["coord"]
        offset = data_dict["offset"]
        center_pred = coord + bias_pred
        center_pred /= self.voxel_size
        logit_pred = F.softmax(logit_pred, dim=-1)
        segment_pred = torch.max(logit_pred, 1)[1]  # [n]
        # cluster
        mask = (
            ~torch.concat(
                [
                    (segment_pred == index).unsqueeze(-1)
                    for index in self.segment_ignore_index
                ],
                dim=1,
            )
            .sum(-1)
            .bool()
        )

        if mask.sum() == 0:
            proposals_idx = torch.zeros(0).int()
            proposals_offset = torch.zeros(1).int()
        else:
            center_pred_ = center_pred[mask]
            segment_pred_ = segment_pred[mask]

            batch_ = offset2batch(offset)[mask]
            offset_ = nn.ConstantPad1d((1, 0), 0)(batch2offset(batch_))
            idx, start_len = ballquery_batch_p(
                center_pred_,
                batch_.int(),
                offset_.int(),
                self.cluster_thresh,
                self.cluster_closed_points,
            )
            proposals_idx, proposals_offset = bfs_cluster(
                segment_pred_.int().cpu(),
                idx.cpu(),
                start_len.cpu(),
                self.cluster_min_points,
            )
            proposals_idx[:, 1] = (
                mask.nonzero().view(-1)[proposals_idx[:, 1].long()].int()
            )

        # get proposal
        proposals_pred = torch.zeros(
            (proposals_offset.shape[0] - 1, center_pred.shape[0]), dtype=torch.int
        )
        proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
        instance_pred = segment_pred[
            proposals_idx[:, 1][proposals_offset[:-1].long()].long()
        ]
        proposals_point_num = proposals_pred.sum(1)
        proposals_mask = proposals_point_num > self.cluster_propose_points
        proposals_pred = proposals_pred[proposals_mask]
        instance_pred = instance_pred[proposals_mask]

        pred_scores = []
        pred_classes = []
        pred_masks = proposals_pred.detach().cpu()
        for proposal_id in range(len(proposals_pred)):
            segment_ = proposals_pred[proposal_id]
            confidence_ = logit_pred[
                segment_.bool(), instance_pred[proposal_id]
            ].mean()
            object_ = instance_pred[proposal_id]
            pred_scores.append(confidence_)
            pred_classes.append(object_)
        if len(pred_scores) > 0:
            pred_scores = torch.stack(pred_scores).cpu()
            pred_classes = torch.stack(pred_classes).cpu()
        else:
            pred_scores = torch.tensor([])
            pred_classes = torch.tensor([])

        return_dict = dict(
            pred_scores=pred_scores,
            pred_masks=pred_masks,
            pred_classes=pred_classes,
            bias_pred=bias_pred,
            seg_logits=logit_pred,
        )
        return return_dict

    def pred_feat_test(self, logit_pred, bias_pred, data_dict):
        """
        - 对非 ignore 类执行基于球查询 + BFS 的实例提案
        - 对于 segment_ignore_index 中的语义类（跳过 -1），为每个 batch 追加一个“整类即实例”的 proposal
        """
        coord = data_dict["coord"]
        offset = data_dict["offset"]  # 假定为前缀和，如 [n1, n1+n2, ...]，长度 = batch_size

        # 1) 预处理
        center_pred = coord + bias_pred
        center_pred /= self.voxel_size
        logit_pred = F.softmax(logit_pred, dim=-1)  # [N, C]
        segment_pred = torch.max(logit_pred, 1)[1]  # [N]
        N = int(segment_pred.numel())
        num_classes = int(logit_pred.shape[-1])

        # 2) 仅对非 ignore 类做聚类
        if getattr(self, "segment_ignore_index", None) and len(self.segment_ignore_index) > 0:
            ignore_masks = []
            for idx in self.segment_ignore_index:
                ignore_masks.append((segment_pred == int(idx)).unsqueeze(-1))
            ignore_cat = torch.concat(ignore_masks, dim=1)  # [N, k]
            mask = ~ignore_cat.sum(-1).bool()  # [N]
        else:
            mask = torch.ones_like(segment_pred, dtype=torch.bool)

        if mask.sum() == 0:
            # 保持后续索引安全：空 proposals_idx 也用 (0, 2) 形状
            proposals_idx = torch.zeros((0, 2), dtype=torch.int)
            proposals_offset = torch.zeros(1, dtype=torch.int)
        else:
            center_pred_ = center_pred[mask]
            segment_pred_ = segment_pred[mask]
            batch_ = offset2batch(offset)[mask]
            offset_ = nn.ConstantPad1d((1, 0), 0)(batch2offset(batch_))

            idx, start_len = ballquery_batch_p(
                center_pred_,
                batch_.int(),
                offset_.int(),
                self.cluster_thresh,
                self.cluster_closed_points,
            )
            proposals_idx, proposals_offset = bfs_cluster(
                segment_pred_.int().cpu(),
                idx.cpu(),
                start_len.cpu(),
                self.cluster_min_points,
            )
            # 将 masked 索引映射回全局点索引
            if proposals_idx.numel() > 0:
                global_indices = torch.nonzero(mask, as_tuple=False).view(-1)
                proposals_idx[:, 1] = global_indices[proposals_idx[:, 1].long()].int()

        # 3) 生成 proposals 二值掩码（行：proposal，列：点）
        proposals_pred = torch.zeros(
            (proposals_offset.shape[0] - 1, N), dtype=torch.int
        )  # 保持在 CPU
        if proposals_idx.numel() > 0:
            proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

        # 每个 proposal 的语义类别（取其首点所在的语义）
        if proposals_offset.shape[0] > 1:
            instance_pred = segment_pred[
                proposals_idx[:, 1][proposals_offset[:-1].long()].long()
            ]
        else:
            instance_pred = torch.tensor([], dtype=segment_pred.dtype)

        # 4) 过滤掉点数过少的 proposal
        if proposals_pred.numel() > 0:
            proposals_point_num = proposals_pred.sum(1)
            proposals_mask = proposals_point_num > self.cluster_propose_points
            proposals_pred = proposals_pred[proposals_mask]
            instance_pred = instance_pred[proposals_mask] if instance_pred.numel() > 0 else instance_pred

        # 5) NEW: 为 ignore 的语义类（跳过 -1）各补一个“整类即实例”的 proposal（按 batch 拆分）
        if getattr(self, "segment_ignore_index", None) and len(self.segment_ignore_index) > 0:
            valid_singletons = []
            for c in self.segment_ignore_index:
                ci = int(c)
                if ci == -1:
                    continue
                if 0 <= ci < num_classes:
                    valid_singletons.append(ci)

            if len(valid_singletons) > 0:
                extra_rows = []
                extra_classes = []

                # 确保 offset 可迭代且为前缀和下标
                start = 0
                for end in offset:
                    end = int(end)
                    for cls_id in valid_singletons:
                        m_local = (segment_pred[start:end] == cls_id)  # (end-start,)
                        if m_local.any():
                            # 构造全局长度的掩码，并仅在本 batch 段内置 1
                            m_global = torch.zeros(N, dtype=torch.int)
                            m_global[start:end] = m_local.detach().cpu().int()
                            extra_rows.append(m_global.unsqueeze(0))  # [1, N]
                            extra_classes.append(cls_id)
                    start = end

                if len(extra_rows) > 0:
                    extra_rows = torch.cat(extra_rows, dim=0)  # [K, N]

                    # —— 与下游保持一致性：这里选择“放到最前面”，则两边都放最前 —— #
                    if proposals_pred.numel() > 0:
                        proposals_pred = torch.cat([extra_rows, proposals_pred], dim=0)
                    else:
                        proposals_pred = extra_rows  # 只有整类实例

                    if instance_pred.numel() > 0:
                        instance_pred = instance_pred.to("cpu")
                        extra_classes_t = torch.tensor(extra_classes, dtype=instance_pred.dtype)
                        instance_pred = torch.cat([extra_classes_t, instance_pred], dim=0)
                    else:
                        instance_pred = torch.tensor(extra_classes, dtype=torch.long)

                    # 与聚类提案一致：再做一次最小点数过滤（两端同掩码）
                    keep = (proposals_pred.sum(1) > self.cluster_propose_points)
                    proposals_pred = proposals_pred[keep]
                    instance_pred = instance_pred[keep]

        # 一致性自检
        assert proposals_pred.shape[0] == instance_pred.shape[0] if instance_pred.numel() > 0 else proposals_pred.shape[
                                                                                                       0] == 0
        assert proposals_pred.shape[1] == N

        # 6) 计算每个 proposal 的置信度与类别
        pred_scores = []
        pred_classes = []
        pred_masks = proposals_pred.detach().cpu()  # [P, N] (CPU, int{0,1})

        for proposal_id in range(len(proposals_pred)):
            seg_mask_cpu = proposals_pred[proposal_id]  # CPU int
            # 避免设备不一致：把掩码移动到 logit_pred 的 device 再做布尔索引
            seg_mask = seg_mask_cpu.to(logit_pred.device).bool()
            cls_id = int(instance_pred[proposal_id]) if instance_pred.numel() > 0 else 0

            # 防御空掩码（理论上已过滤）
            if seg_mask.any():
                confidence_ = logit_pred[seg_mask, cls_id].mean()
            else:
                confidence_ = torch.tensor(0.0, device=logit_pred.device)

            pred_scores.append(confidence_.detach().cpu())
            pred_classes.append(cls_id)

        if len(pred_scores) > 0:
            pred_scores = torch.stack(pred_scores).cpu()
            pred_classes = torch.tensor(pred_classes, dtype=torch.long).cpu()
        else:
            pred_scores = torch.tensor([])
            pred_classes = torch.tensor([])

        return dict(
            pred_scores=pred_scores,  # [P]
            pred_masks=pred_masks,  # [P, N] (CPU, int{0,1})
            pred_classes=pred_classes,  # [P] (CPU, long)
            bias_pred=bias_pred,
            seg_logits=logit_pred,  # softmax 后
        )

    def forward(self, data_dict):
        feat = self.backbone(data_dict)
        # feat = Point.feat
        bias_pred = self.bias_head(feat)
        logit_pred = self.seg_head(feat)
        return_dict = dict()
        if not self.test:
            return_dict = self.loss(logit_pred, bias_pred, data_dict)
        if not self.training:
            if self.test:
                pred_dict = self.pred_feat_test(logit_pred, bias_pred, data_dict)
            else:
                pred_dict = self.pred_feat_test(logit_pred, bias_pred, data_dict)
            return_dict.update(pred_dict)
        return return_dict

