## Result and models
### Preprocessing: Org + CLAHE + CLAHE
The result is evaluated by testing on fold1 and keep default learning rate scheadule. 
Training arguments:
1. seed 42
2. deterministic
3. auto-scale-lr
Check config files for more details.

| Architecture |    Backbone     | Lr schd | #Params (M) | #FLOPs (G) | $AP_{50}$ |
| :----------: | :-----: | :-----: | :------: | :------------: | :---------: | 
| Faster R-CNN | R-50 | 2x | 41.12 | 206.66 | 0.506 | 
| RetinaNet | R-50 | 2x | 36.1 | 204.36 | 0.527 |
| Cascade R-CNN | R-50 | 20e | 68.93 | 234.46 | 0.510 | 
| DETR | R-50 | 150e | 41.28 | 91.62 | 0.527 | 
| VFNet | R-50 | 2x | 32.48 | 188.97 | 0.568 | 
| Deformable DETR | R-50 | 50e | 39.82 | 195.23 | 0.530 |
| TOOD | R-50 | 2x | 31.79 | 180.51 | 0.476 | 