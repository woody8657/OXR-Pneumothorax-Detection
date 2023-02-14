## Result and models
### Preprocessing: Org + CLAHE + CLAHE
The result is evaluated by testing on fold1 and keep default learning rate scheadule. 
Training arguments:
1. 100 epochs
2. 16 batch size
3. 640 input size
Check config files for more details.

| Architecture | #Params (M) | #FLOPs (G) | $AP_{50}$ |
| :----------: || :------: | :------------: | :---------: | 
| YOLOv5 | 7.0 | 15.8 | 0.467 | 
| YOLOv6 | 18.5 | 45.2 | 0.566 |
| YOLOv7 | 36.5 | 103.2 | 0.200 | 
| YOLOv8 | 11.1 | 28.4 | 0.609 | 