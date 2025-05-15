
# configs/visdrone_retinanet.py
_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Dataset settings
data_root = 'data/VisiDrone/'
metainfo = {
    'classes': ('pedestrian', 'person', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
        (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
        (0, 0, 192), (250, 170, 30)
    ]
}

# device = "cuda"

# Model configuration (CORRECTED)
# model = dict(
#     bbox_head=dict(
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             octave_base_scale=4,  # Base scale for octave
#             scales_per_octave=3,  # Number of scales per octave
#             ratios=[0.5, 1.0, 2.0],
#             strides=[8, 16, 32, 64, 128])
#     )
# )

model = dict(
    bbox_head=dict(
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            activated=False  # ← Critical fix for CUDA implementation
        ),
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,  # Base scale for octave
            scales_per_octave=3,  # Number of scales per octave
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128])
    )
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn'),  # 'fork' on Linux
    dist_cfg=dict(backend='nccl')
)

# Training configuration
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train_coco.json',
        data_prefix=dict(img='train/images/'))
)

# To force GPU usage, use this launch command:
# python tools/train.py configs/visdrone_retinanet.py --device cuda:0