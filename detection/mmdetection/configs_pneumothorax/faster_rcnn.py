# inherited configuration
_base_ = [
    '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
]

# model settings, change num_classes to fit our task
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1
            )))

# data path
classes = ('Pneumothorax',)
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        img_prefix='/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/images',
        classes=classes,
        ann_file='/home/u/woody8657/projs/OXR-Pneumothorax-Detection/data/annotation_coco_format/train_fold1.json'),
    val=dict(
        img_prefix='/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/images',
        classes=classes,
        ann_file='/home/u/woody8657/projs/OXR-Pneumothorax-Detection/data/annotation_coco_format/test_fold1.json'),
    test=dict(
        img_prefix='/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/images',
        classes=classes,
        ann_file='/home/u/woody8657/projs/OXR-Pneumothorax-Detection/data/annotation_coco_format/holdout.json'))

# criteria of saving model
checkpoint_config = dict(interval=24)
evaluation = dict(interval=1, save_best='bbox_mAP_50')

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

# directory of training log and model weights
work_dir = './work_dirs/'

# pre-trained model
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
