# ground truth
python plot_bbox.py \
    --bbox-json ../../../data/annotation_coco_format/holdout.json \
    --img-prefix /data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/images_org/ \
    --query-imgs P214260000945.png P214260001894.png \
    --conf-threshold 0.2 \
    --save-dir ../visulization/gt_images

# prediction
python plot_bbox.py \
    --ref-json ../../../data/annotation_coco_format/holdout.json \
    --bbox-json ../prediction.bbox.json \
    --img-prefix /data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/images_org/ \
    --query-imgs P214260000945.png P214260001894.png \
    --conf-threshold 0.2 \
    --save-dir ../visulization/prediction_images


