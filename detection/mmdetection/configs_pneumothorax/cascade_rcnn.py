# inherited configuration
_base_ = [
    '../configs/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco.py'
]

# model settings, change num_classes to fit our task

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=1
            ),
             dict(
                type='Shared2FCBBoxHead',
                num_classes=1
            ),
             dict(
                type='Shared2FCBBoxHead',
                num_classes=1
            )
        ]))

# data path
classes = ('Pneumothorax',)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        img_prefix='/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/images_unsharp_mask',
        classes=classes,
        ann_file='/home/u/woody8657/projs/OXR-Pneumothorax-Detection/data/annotation_coco_format/train_fold1.json'),
    val=dict(
        img_prefix='/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/images_unsharp_mask',
        classes=classes,
        ann_file='/home/u/woody8657/projs/OXR-Pneumothorax-Detection/data/annotation_coco_format/test_fold1.json'),
    test=dict(
        img_prefix='/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/images_unsharp_mask',
        classes=classes,
        ann_file='/home/u/woody8657/projs/OXR-Pneumothorax-Detection/data/annotation_coco_format/holdout.json'))

# criteria of saving model
checkpoint_config = dict(interval=-1)
evaluation = dict(interval=1, save_best='bbox_mAP_50')

# directory of training log and model weights
work_dir = '/home/u/woody8657/data/C426_Pneumothorax_grid/Reproduce/unsharp_mask/cascade_rcnn'

# pre-trained model
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth'