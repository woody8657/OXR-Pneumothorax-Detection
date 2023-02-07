docker run -it --rm \
    --gpus all \
    --runtime=nvidia \
    --shm-size=128g \
    --name oxr_docker_pneumothorax \
    -v $(pwd)/src_det:/mmdetection/src \
    -v ${1:-$(pwd)/../data/inputs}:/tmp/inputs \
    -v ${2:-$(pwd)/../data/outputs}:/tmp/outputs \
    oxr_docker_pneumothorax_det \
    bash src/detect.sh ${3:-0}

   
    
