# Pneumothorax Detection
## Install YOLOv5~v8
1. Create a virtual environment
```
conda create --name pneumothorax_yolo python=3.8 -y
```
2. Activate it
```
conda activate pneumothorax_yolo
```
3. Install requitements of YOLOv5-v8, the package version is compatible between v58v8. Hence, it's fine to install them in single virtual environment.
```
cd ./detection/yolo_series/YOLOv5/
   ./detection/yolo_series/YOLOv6/
   ./detection/yolo_series/YOLOv7/
   ./detection/yolo_series/YOLOv8/
pip install -r requirements.txt
```
## Result and models
### Preprocessing: Org + CLAHE + CLAHE
The result is evaluated by testing on fold1 and keep default learning rate scheadule. 
Training arguments:
1. 100 epochs
2. 16 batch size
3. 640 input size
Check config files for more details.

| Architecture | #Params (M) | #FLOPs (G) | $AP_{50}$ |
| :----------: | :------: | :------------: | :---------: | 
| YOLOv5 | 7.0 | 15.8 | 0.467 | 
| YOLOv6 | 18.5 | 45.2 | 0.566 |
| YOLOv7 | 36.5 | 103.2 | 0.200 | 
| YOLOv8 | 11.1 | 28.4 | 0.609 | 