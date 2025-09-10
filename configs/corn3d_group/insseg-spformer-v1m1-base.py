_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 4  # bs: total bs in all gpus
num_worker = 2
mix_prob = 0
empty_cache = True
enable_amp = True
evaluate = True

class_names = [
    "maize",
]
num_class = 1
segment_ignore_index = [-1]

# model settings
model = dict(
    type="SPFormer-v1m1",
    pre_backbone=dict(
        type="SpUNet-v1m1",
        in_channels=6,
        num_classes=0,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    pre_backbone_out_channels=96,
    num_class=num_class,
    pool='mean',
    decoder=dict(
        num_layer=6,
        num_query=400,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        pe=False,
    ),
    criterion=dict(
        loss_weight=[0.0, 1.0, 1.0, 0.5],
        cost_weight=[0.5, 1.0, 1.0],
        non_object_weight=0.1
    ),
    test_cfg=dict(
        topk_insts=100,
        score_thr=0.0,
        npoint_thr=100,
    ),
    norm_eval=False,
)

# scheduler settings
epoch = 200
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.05)
scheduler = dict(type="PolyLR")

# dataset settings
dataset_type = "Corn3dGroupDataset"
data_root = "data/corn3d_group/20240301_cs_group"

data = dict(
    num_classes=num_class,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.1),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "normal", "segment", "instance", "superpoint"),
            ),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            # dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "scene_id",
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "instance_centroid",
                    "bbox",
                    "superpoint"
                ),
                feat_keys=("coord", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "normal", "segment", "instance", "superpoint"),
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "scene_id",
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                    "superpoint"
                ),
                feat_keys=("coord", "normal"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(),  # currently not available
)

hooks = [
    dict(type="CheckpointLoader", keywords="backbone.", replacement="pre_backbone."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="OnlyInsSegEvaluator",
        segment_ignore_index=segment_ignore_index,
        instance_ignore_index=-1,
    ),
    dict(type="CheckpointSaver", save_freq=None),
]
