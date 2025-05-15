_base_ = [
    '../configs/_base_/models/retinanet_r50_fpn.py',
    '../configs/_base_/datasets/coco_detection.py',
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]

model = dict(
    type='MyCustomDetector',
    backbone=dict(...),
    neck=dict(...),
    bbox_head=dict(...)
)