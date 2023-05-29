# Run whole dir
configs_dir=./configs_pneumothorax_10f
for entry in "$configs_dir"/*
do
  CUDA_VISIBLE_DEVICES=0,1  ./tools/dist_train.sh $entry 2 \
        --auto-scale-lr \
        --seed 42 \
        --deterministic
done