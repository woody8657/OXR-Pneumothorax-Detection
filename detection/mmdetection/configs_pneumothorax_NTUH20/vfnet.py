_base_ = '../configs/vfnet/vfnet_r50_fpn_1x_coco.py'

model = dict(
    bbox_head=dict(num_classes=1))


classes = ('Pneumothorax',)
data = dict(
    test=dict(
        img_prefix='../../data/images_NTUH_20',
        classes=classes,
        ann_file='../../data/10f_traininglist/NTUH_20_gt.json')
    )
