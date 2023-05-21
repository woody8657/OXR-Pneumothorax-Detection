# inherited configuration
_base_ = [
    '../configs/retinanet/retinanet_r50_fpn_2x_coco.py'
]

# model settings, change num_classes to fit our task
model = dict(
    bbox_head=dict(
        num_classes=1
    ))

# data path
classes = ('Pneumothorax',)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        img_prefix='../../data/images',
        classes=classes,
        ann_file='../../data/annotation_coco_format/train_fold1.json'),
    val=dict(
        img_prefix='../../data/images',
        classes=classes,
        ann_file='../../data/annotation_coco_format/test_fold1.json'),
    test=dict(
        img_prefix='../../data/images',
        classes=classes,
        ann_file='../../data/annotation_coco_format/holdout.json'))

# criteria of saving model
checkpoint_config = dict(interval=-1)
evaluation = dict(interval=1, save_best='bbox_mAP_50')

# directory of training log and model weights
work_dir = '/data2/smarted/PXR/data/C426_Pneumothorax_grid/pneumothorax_detection'

# pre-trained model
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'