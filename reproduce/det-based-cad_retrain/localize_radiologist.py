import os
import json
import numpy as np
import scipy.stats as st
import skimage.measure
from tqdm import tqdm
import cv2
import random

def set_random_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed) 

def grid_bbox2mask(json_path):
    tmp = json.load(open(json_path))
    mask = np.zeros((tmp['size'][1], tmp['size'][0]))
    for j in range(len(tmp['shapes'])):
        x_a, y_a = tmp['shapes'][j]['a']
        x_b, y_b = tmp['shapes'][j]['b']
        x, y, w, h = int(min(x_a, x_b)), int(min(y_a, y_b)), int(max(x_a, x_b)-min(x_a, x_b)), int(max(y_a, y_b)-min(y_a, y_b))        
        mask[y:y+h,x:x+w] = 1
    
    return mask

def remove_nan(input):
    output = []
    for i in input:
        if i > 0:
            output.append(i)
    return output

def circumscribed(mask, return_area=False):
    mask = skimage.measure.label(mask,connectivity=1)
    xyxy_bbox = np.empty((0,4), int)
    
    for region in skimage.measure.regionprops(mask):
        minr, minc, maxr, maxc = region.bbox
        xyxy_bbox = np.append(xyxy_bbox, np.expand_dims(np.array([minc, minr, maxc, maxr]), axis=0), axis=0)

    xywh_bbox = []
    area = []
    for i in range(xyxy_bbox.shape[0]):
        (x1,y1) = map(int,(xyxy_bbox[i,0], xyxy_bbox[i,1]))
        (x2,y2) = map(int,(xyxy_bbox[i,2], xyxy_bbox[i,3]))
        xywh_bbox.append([x1, y1, x2-x1, y2-y1])
        area.append((x2-x1)*(y2-y1))
    if return_area:
        return max(area)
        

    circumscribed_mask = np.zeros(mask.shape)    
    for x,y,w,h in xywh_bbox:
        circumscribed_mask[y:y+h,x:x+w] = 1

    return circumscribed_mask


def statistic(data):
    print(np.mean(data),',' , np.std(data, ddof=1), (np.percentile(data, 2.5), np.percentile(data, 97.5)))


if __name__ == '__main__':
    gt_path = '../../data/classification/labels_NTUH_20'
    pid_list = os.listdir(gt_path)
    with open('./prediction/pred_overall_tp_2cls_list.json', 'r') as file:
        pid_list = json.load(file)
    set_random_seed()
    thres_33 = 0.0318689602574198
    thres_66 = 0.0804696718241336
    size_LUT = {'s': [], 'm': [], 'l': []}
    for pid in pid_list:
        annotations_list = os.listdir(os.path.join(gt_path, pid))
        if len(annotations_list) == 1:
            img = cv2.imread(os.path.join('../../data/images_NTUH_20', pid+'.png'))
            mask = grid_bbox2mask(os.path.join(gt_path, pid, 'label.json'))
            num_pixel = len(np.where(mask==1)[0])/(img.shape[0]*img.shape[1])
            if num_pixel < thres_33:
                size_LUT['s'].append(pid)
            elif num_pixel >= thres_33 and num_pixel < thres_66:
                size_LUT['m'].append(pid)
            elif num_pixel >= thres_66:
                size_LUT['l'].append(pid)
            else:
                raise
    # raise
    # with open('size_LUT_G301.json', 'w') as file:
    #     json.dump(size_LUT, file)
    annotation_path = '/data2/smarted/PXR/data/C426_Pneumothorax_grid/G3_01_annotation_pid/'
    # pid_list = os.listdir(annotation_path)
   
    tp_iou = []
    tp_iou_s = []
    tp_iou_m = []
    tp_iou_l = []
    tp_dice = []
    tp_dice_s = []
    tp_dice_m = []
    tp_dice_l = []
    performance_detail = []
    for pid in tqdm(pid_list):
        if pid == 'annotation_count_G3_01.json':
            continue
        annotations_list = os.listdir(os.path.join(annotation_path, pid))
        if pid in size_LUT['s'] or pid in size_LUT['m'] or pid in size_LUT['l']:
            if len(annotations_list) == 2:
                json1_path = os.path.join(annotation_path, pid, annotations_list[0])
                json2_path = os.path.join(annotation_path, pid, annotations_list[1])
                mask1 = circumscribed(grid_bbox2mask(json1_path))
                mask2 = circumscribed(grid_bbox2mask(json2_path))
                # mask1 = grid_bbox2mask(json1_path)
                # mask2 = grid_bbox2mask(json2_path)
            
                tmp_mask = mask1 + mask2
                iou = len(np.where(tmp_mask==2)[0])/len(np.where(tmp_mask>=1)[0])
                dice = 2*len(np.where(tmp_mask==2)[0])/(len(np.where(mask1==1)[0])+len(np.where(mask2==1)[0]))
                tp_iou.append(iou)      
                tp_dice.append(dice)
                if pid in size_LUT['s']:
                    tp_iou_s.append(iou)
                    tp_dice_s.append(dice)
                    performance_detail.append({'iou': iou, 'dice': dice, 'size': 's'})
                elif pid in size_LUT['m']:
                    tp_iou_m.append(iou)
                    tp_dice_m.append(dice)
                    performance_detail.append({'iou': iou, 'dice': dice, 'size': 'm'})
                elif pid in size_LUT['l']:
                    tp_iou_l.append(iou)
                    tp_dice_l.append(dice)
                    performance_detail.append({'iou': iou, 'dice': dice, 'size': 'l'})
                else:
                    raise


    iteration = 1000
    tp_iou_bootsrapping = []
    tp_iou_s_bootsrapping = []
    tp_iou_m_bootsrapping = []
    tp_iou_l_bootsrapping = []
    tp_dice_bootsrapping = []
    tp_dice_s_bootsrapping = []
    tp_dice_m_bootsrapping = []
    tp_dice_l_bootsrapping = []
    for i in tqdm(range(iteration)):
        tp_iou = []
        tp_iou_s = []
        tp_iou_m = []
        tp_iou_l = []
        tp_dice = []
        tp_dice_s = []
        tp_dice_m = []
        tp_dice_l = []
        for id in np.random.randint(len(performance_detail), size=len(performance_detail)):
            iou = performance_detail[id]['iou']
            dice = performance_detail[id]['dice']
            size = performance_detail[id]['size']
            tp_iou.append(iou)
            tp_dice.append(dice)
            if size == 's':
                tp_iou_s.append(iou)
                tp_dice_s.append(dice)
            elif size == 'm':
                tp_iou_m.append(iou)
                tp_dice_m.append(dice)
            elif size == 'l':
                tp_iou_l.append(iou)
                tp_dice_l.append(dice)
            else:
                raise 
        tp_iou_bootsrapping.append(np.mean(tp_iou))
        tp_iou_s_bootsrapping.append(np.mean(tp_iou_s))
        tp_iou_m_bootsrapping.append(np.mean(tp_iou_m))
        tp_iou_l_bootsrapping.append(np.mean(tp_iou_l))
        tp_dice_bootsrapping.append(np.mean(tp_dice))
        tp_dice_s_bootsrapping.append(np.mean(tp_dice_s))
        tp_dice_m_bootsrapping.append(np.mean(tp_dice_m))
        tp_dice_l_bootsrapping.append(np.mean(tp_dice_l))         

    print(f"Pneumothorax size: overall")
    statistic(tp_dice_bootsrapping)
    print(f"Pneumothorax size: l")
    statistic(tp_dice_l_bootsrapping)
    print(f"Pneumothorax size: m")
    statistic(tp_dice_m_bootsrapping)
    print(f"Pneumothorax size: s")
    statistic(remove_nan(tp_dice_s_bootsrapping))
    
    

    tmp = {'all': tp_dice_bootsrapping, 's': remove_nan(tp_dice_s_bootsrapping), 'm': remove_nan(tp_dice_m_bootsrapping), 'l': remove_nan(tp_dice_l_bootsrapping)}
    
    with open('./prediction/dice_annotator.json', 'w') as file:
        json.dump(tmp, file)