# The log directory you assign in config file
config=$1
log_dir=$2
outputname=$3

CUDA_VISIVLE_DECIVES=5 python ./tools/test.py \
    $config \
    $log_dir/best* \
    --eval-options jsonfile_prefix=$outputname \
    --eval bbox  
