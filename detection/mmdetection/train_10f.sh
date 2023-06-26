Run whole dir
configs_dir=./configs_pneumothorax_10f
for entry in "$configs_dir"/*
do
  CUDA_VISIBLE_DEVICES=0,1,2  ./tools/dist_train.sh $entry 3 \
        --auto-scale-lr \
        --seed 42 \
        --deterministic
done

configs_dir=./configs_pneumothorax_10f_noholdout
for entry in "$configs_dir"/*
do
  CUDA_VISIBLE_DEVICES=5,6,7  ./tools/dist_train.sh $entry 3 \
        --auto-scale-lr \
        --seed 42 \
        --deterministic
done

