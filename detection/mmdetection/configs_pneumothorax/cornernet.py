# inherited configuration
_base_ = [
    '../configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py'
]

# model settings, change num_classes to fit our task
model = dict(
    bbox_head=dict(
        num_classes=1
        ))

# data path
classes = ('Pneumothorax',)
data = dict(
    samples_per_gpu=2,
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
work_dir = '/home/u/woody8657/data/C426_Pneumothorax_grid/Reproduce/unsharp_mask/cornernet'

# pre-trained model
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco/cornernet_hourglass104_mstest_8x6_210e_coco_20200825_150618-79b44c30.pth'