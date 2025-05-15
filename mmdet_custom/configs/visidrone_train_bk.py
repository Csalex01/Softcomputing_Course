# import os

# os.environ['USE_LIBUV'] = '0'  # Disable libuv before importing torch.distributed

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

device = "cuda:0"


env_cfg = dict(
    cudnn_benchmark=True,  # Enable cuDNN auto-tuner
    mp_cfg=dict(
        mp_start_method='spawn',  # 'spawn' on Windows
    ),
    dist_cfg=dict(backend='nccl')  # NVIDIA Collective Comm Library
)


# env_cfg = dict(
#     mp_cfg=dict(mp_start_method='spawn'),  # Required for Windows
#     dist_cfg=dict(backend='gloo')  # Alternative to NCCL if issues occur
# )

train_dataloader = dict(
    batch_size=4,
    num_workers=20,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train_coco.json',
        data_prefix=dict(img='train/images/')
    )
)

# Correct anchor generator configuration
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
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        device='cuda:0',  # Force all data to GPU
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        pad_size_divisor=32
    ),
    backbone=dict(
        with_cp=True  # Gradient checkpointing to save memory
    ),
    bbox_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]
        )
    )
)

# Enable text logging to console
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),  # Log every 10 iterations
)

# # In your config file (e.g., rtmdet_tiny_visdrone.py)
# visualizer = dict(
#     type='DetLocalVisualizer',
#     vis_backends=[
#         dict(type='LocalVisBackend'),
#         dict(type='TensorboardVisBackend',  # Must be included
#              save_dir='work_dirs/training')  # Match --work-dir
#     ],
#     name='visualizer'
# )