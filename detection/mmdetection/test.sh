# The log directory you assign in config file
log_dir=$1


CUDA_VISIVLE_DECIVES=1 python ./tools/test.py \
    $log_dir/*.py \
    $log_dir/best* \
    --eval-options jsonfile_prefix=./prediction \
    --eval bbox 
    