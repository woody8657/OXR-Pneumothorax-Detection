_base_ = '../configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=1
    )
)

classes = ('Pneumothorax',)
data = dict(
    test=dict(
        img_prefix='../../data/images_NTUH_20',
        classes=classes,
        ann_file='../../data/10f_traininglist/NTUH_20_gt.json')
    )
