CUDA_VISIBLE_DEVICES=1,2,3 ./tools/dist_train.sh ./configs_pneumothorax/faster_rcnn.py 3 \
    --work-dir work_dirs/faster_rcnn \
    --auto-scale-lr \
    --seed 42 \
    --deterministic