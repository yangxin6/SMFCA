_base_ = ["../../_base_/default_runtime.py"]

# misc custom setting
batch_size = 4  # bs: total bs in all gpus
num_worker = 4
mix_prob = 0
empty_cache = False
enable_amp = True
evaluate = True

class_names = [
    'none',
    "ground",
    "vegetation",
    "car",
    "truck",
    "powerline",
    "fence",
    "pole",
    "buildings",
]
num_classes = 9
segment_ignore_index = 0

# model settings


model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="SpVUNet-v1m2",
        in_channels=3,
        num_classes=num_classes,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    # criteria=[dict(type="CrossEntropyLoss", weight=[0, 0.23, 0.34, 15.86, 54.7, 51.21, 27.08, 147.92, 0.72], loss_weight=1.0, ignore_index=segment_ignore_index)],
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=segment_ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=segment_ignore_index),
    ],
)

# scheduler settings
epoch = 100
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.00001, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.00001, 0.000001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.000001)]


# dataset settings
dataset_type = "Corn3dGroupSemanticDataset"
data_root = "data/corn3d_group_semantic/fast/exp_datasets/dales_til_8_pth"
data_test = "data/corn3d_group_semantic/fast/exp_datasets/dales_pth"

data = dict(
    num_classes=num_classes,
    ignore_index=segment_ignore_index,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "segment")
            ),
            dict(type="CenterShift", apply_z=False),
            # dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "segment")
            ),
            dict(type="CenterShift", apply_z=False),
            # dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_test,
        transform=[
            dict(type="CenterShift", apply_z=True),
            # dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="test",
                keys=["coord"],
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=["coord"],
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
    ),
)


hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]
