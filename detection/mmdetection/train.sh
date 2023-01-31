# Declare the model list
config_dir=./configs_pneumothorax/
ModelList=("cascade_rcnn"  "deformable_detr"  "detr"  "faster_rcnn"  "retinanet"  "tood"  "vfnet"   "cornernet")
for model in ${ModelList[*]}; do
    CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh $config_dir$model.py 2 \
        --auto-scale-lr \
        --seed 42 \
        --deterministic
done

# Run whole dir
# search_dir=./configs_pneumothorax
# for entry in "$search_dir"/*
# do
#   CUDA_VISIBLE_DEVICES=0,1,2,3  ./tools/dist_train.sh $entry 4 \
#         --auto-scale-lr \
#         --seed 42 \
#         --deterministic
# done