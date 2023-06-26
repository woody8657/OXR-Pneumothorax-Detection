python ../tools/test.py \
    /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/deformable_detr_G301.py \
    /home/u/woody8657/data/C426_Pneumothorax_grid/landing/docker_new_10f/deformable_detr/fold_2/best* \
    --eval-options jsonfile_prefix=./inference_json/deformable_detr_f2 \
    --eval bbox 

python ../tools/test.py \
    /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/deformable_detr_G301.py \
    /home/u/woody8657/data/C426_Pneumothorax_grid/landing/docker_new_10f/deformable_detr/fold_4/best* \
    --eval-options jsonfile_prefix=./inference_json/deformable_detr_f4 \
    --eval bbox 

python ../tools/test.py \
    /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/tood_G301.py \
    /home/u/woody8657/data/C426_Pneumothorax_grid/landing/docker_new_10f/tood/fold_7/best* \
    --eval-options jsonfile_prefix=./inference_json/tood_f7 \
    --eval bbox 

python ../tools/test.py \
    /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/tood_G301.py \
    /home/u/woody8657/data/C426_Pneumothorax_grid/landing/docker_new_10f/tood/fold_2_positive/best* \
    --eval-options jsonfile_prefix=./inference_json/tood_positive_f2 \
    --eval bbox 

python ../tools/test.py \
    /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/vfnet_G301.py \
    /home/u/woody8657/data/C426_Pneumothorax_grid/landing/docker_new_10f/vfnet/fold_8_positive/best* \
    --eval-options jsonfile_prefix=./inference_json/vfnet_positive_f8 \
    --eval bbox 