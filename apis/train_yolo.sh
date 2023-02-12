# cd ../detection/yolo_series/YOLOv5/
# CUDA_VISIBLE_DEVICES=0 python train.py --data PTX.yaml --weights https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt --batch 16
# CUDA_VISIBLE_DEVICES=0 python train.py --data PTX.yaml --weights https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt --batch 32
# cd ../YOLOv8/
# CUDA_VISIBLE_DEVICES=0 yolo detect train data=PTX.yaml model=yolov8s.pt batch=16
# CUDA_VISIBLE_DEVICES=0 yolo detect train data=PTX.yaml model=yolov8s.pt batch=32
cd ../detection/yolo_series/YOLOv7/
rm /home/u/woody8657/projs/OXR-Pneumothorax-Detection/data/*.cache
rm /home/u/woody8657/projs/OXR-Pneumothorax-Detection/data/yolo_group/*.cache
python train.py --workers 8 --device 1 --batch-size 32 --data data/PTX.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml --project /home/u/woody8657/data/C426_Pneumothorax_grid/Reproduce/org+clahe+clahe/yolov7
python train.py --workers 8 --device 1 --batch-size 16 --data data/PTX.yaml --img 640 640 --epochs 100 --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml --project /home/u/woody8657/data/C426_Pneumothorax_grid/Reproduce/org+clahe+clahe/yolov7
cd ../YOLOv6/
python tools/train.py --batch 32  --conf configs/yolov6s_finetune.py --data data/PTX_v6only.yaml --fuse_ab --device 1 --output-dir /home/u/woody8657/data/C426_Pneumothorax_grid/Reproduce/org+clahe+clahe/yolov6
python tools/train.py --batch 16  --conf configs/yolov6s_finetune.py --data data/PTX_v6only.yaml --epochs 100 --fuse_ab --device 1 --output-dir /home/u/woody8657/data/C426_Pneumothorax_grid/Reproduce/org+clahe+clahe/yolov6