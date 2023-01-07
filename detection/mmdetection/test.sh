# the log directory you assign in config file
log_dir=$1

python tools/test.py $log_dir/*.py $log_dir/best* \
    --eval bbox