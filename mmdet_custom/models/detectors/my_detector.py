from mmdet.registry import MODELS
from mmdet.models.detectors import SingleStageDetector

@MODELS.register_module()
class MyCustomDetector(SingleStageDetector):
    def __init__(self, backbone, neck=None, bbox_head=None, train_cfg=None, test_cfg=None, **kwargs):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg, **kwargs)
    
    def forward(self, inputs, *args, **kwargs):
        # Your custom forward logic
        x = self.extract_feat(inputs)
        return self.bbox_head(x)