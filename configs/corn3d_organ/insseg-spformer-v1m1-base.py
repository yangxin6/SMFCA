_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 20  # bs: total bs in all gpus
num_worker = 10
mix_prob = 0
empty_cache = False
enable_amp = True
evaluate = True

class_names = [
    "stem",
    "xs",
    "cs",
    "leaf",
]
num_class = 4
segment_ignore_index = [-1]

# model settings
model = dict(
    type="SPFormer-v1m1",
    pre_backbone=dict(
        type="SpUNet-v1m1",
        in_channels=6,
        num_classes=0,
        channels=(32, 64, 96, 128, 160, 32),
        layers=(2, 2, 2, 2, 2, 1),
    ),
    pre_backbone_out_channels=32,
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
        class_weight=[0.12, 1.0, 0.37, 0.01],
        loss_weight=[0.5, 1.0, 1.0, 0.5],
        cost_weight=[0.5, 1.0, 1.0],
        non_object_weight=0.001
    ),
    test_cfg=dict(
        topk_insts=100,
        score_thr=0.0,
        npoint_thr=100,
    ),
    norm_eval=False,
)

# scheduler settings
epoch = 400
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
dataset_type = "Corn3dOrganDataset"
data_root = "data/corn3d_organ/20231129_v1"

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
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),  # [-180, 180]
            dict(type="RandomRotate", angle=[-1 / 180, 1 / 180], axis="x", p=0.5),  # [-1, 1]
            dict(type="RandomRotate", angle=[-1 / 180, 1 / 180], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.1),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "color", "normal", "segment", "instance", "superpoint"),
            ),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
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
                    "color",
                    "grid_coord",
                    "segment",
                    "instance",
                    "instance_centroid",
                    "bbox",
                    "superpoint"
                ),
                feat_keys=("color", "normal"),
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
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "color", "normal", "segment", "instance", "superpoint"),
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
                    "color",
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
                feat_keys=("color", "normal"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(),  # currently not available
)

hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="InsSegEvaluator",
        segment_ignore_index=segment_ignore_index,
        instance_ignore_index=-1,
    ),
    dict(type="CheckpointSaver", save_freq=None),
]
