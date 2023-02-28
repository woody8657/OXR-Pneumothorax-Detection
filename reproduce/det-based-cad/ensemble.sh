# python cocoeval.py \
#     --gt /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/training_list/holdout.json \
#     --json ./inference_json/deformable_detr_f2.bbox.json
# python cocoeval.py \
#     --gt /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/training_list/holdout.json \
#     --json ./inference_json/deformable_detr_f4.bbox.json
# python cocoeval.py \
#     --gt /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/training_list/holdout.json \
#     --json ./inference_json/tood_f7.bbox.json
# python cocoeval.py \
#     --gt /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/training_list/holdout.json \
#     --json ./inference_json/tood_positive_f2.bbox.json
# python cocoeval.py \
#     --gt /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/training_list/holdout.json \
#     --json ./inference_json/vfnet_positive_f8.bbox.json



python ensemble.py \
    --gt /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/G3_01/G3_01_mmdet_ver2.json \
    --jsons inference_json/*.json \
    --method wbf  \
    --output-name ensemble.json
