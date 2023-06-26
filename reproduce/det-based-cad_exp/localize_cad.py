import argparse
import json
import numpy as np
import cv2
import scipy.stats as st
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt

def set_random_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)  

def read_json(json_path):
    with open(json_path, 'r') as file:
        output = json.load(file)
    return output

def remove_nan(input):
    output = []
    for i in input:
        if i > 0:
            output.append(i)
    return output


def statistic(data):
    # print(np.mean(data),',' , np.std(data, ddof=1), st.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) )
    print(np.mean(data),',' , np.std(data, ddof=1), (np.percentile(data, 2.5), np.percentile(data, 97.5)))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', default='./prediction/NTUH_20_gt.json', help='ground truth path')
    parser.add_argument('--json', default='./prediction/ensemble_2cls.json', help='prediction')
    opt = parser.parse_args()

    set_random_seed()

    gt = read_json(opt.gt)
    pred = read_json(opt.json)
    with open('./prediction/NTUH_20_size_LUT.json', 'r') as file:
        size_LUT = json.load(file)
    # old version
    threshold = 0.20447373295685223
    
    # threshold = 0.20566484218048947
    gt_positive = []
    # {"id": 0, "image_id": 8, "category_id": 0, "segmentation": [], "area": 457640, "bbox": [1314, 359, 673, 680], "iscrowd": 0}
    for bbox in gt['annotations']:
        gt_positive.append(bbox['image_id'])
    # {"image_id": 0, "bbox": [1082.6519372463226, 457.7657516002655, 583.1782308220863, 326.8219587802887], "score": 0.16599809375030353, "category_id": 0}
    pred_positive = []
    for bbox in pred:
        if bbox['score'] > threshold:
            pred_positive.append(bbox['image_id'])
    
    tp = list(set(gt_positive).intersection(set(pred_positive)))
    tp_pid = [gt['images'][tp[i]]['file_name'][:-4] for i in range(len(tp))]
   
    with open('./prediction/pred_overall_tp_2cls_list.json', 'w') as file:
        json.dump(tp_pid, file)
    tp_iou = []
    tp_iou_s = []
    tp_iou_m = []
    tp_iou_l = []
    tp_dice = []
    tp_dice_s = []
    tp_dice_m = []
    tp_dice_l = []
    class_s = []
    # img_path = '/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/G3_01_preprocessed'
    # save_path = '/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/G3_01_display'
    performance_detail = []
    for id in tqdm(tp):
        # print(id)
        # print(gt['images'][id])
        gt_mask = np.zeros((gt['images'][id]['height'], gt['images'][id]['width']))
        PTX_area = []
        # img = cv2.imread(os.path.join(img_path, gt['images'][id]['file_name']))
        # img[:,:,1] = img[:,:,2] = img[:,:,0]
        for bbox in gt['annotations']:
            if bbox['image_id'] == id:
                x,y,w,h = bbox['bbox']
                gt_mask[y:y+h,x:x+w] = 1
                PTX_area.append(bbox['area'])

        #         if bbox['area'] > 0 and bbox['area'] <= 415*500:
        #             cv2.putText(img, "Small", (int(x), int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        #             cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 5)
        #         elif bbox['area'] > 415*500 and bbox['area'] <= 777*1386:
        #             cv2.putText(img, "Medium", (int(x), int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
        #             cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 5)
        #         elif bbox['area']> 777*1386:
        #             cv2.putText(img, "Large", (int(x), int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
        #             cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 5)
        #         else:
        #             raise
        # cv2.imwrite(os.path.join(save_path, gt['images'][id]['file_name']+'.png'), img)
                

       
        pred_mask = np.zeros((gt['images'][id]['height'], gt['images'][id]['width']))
        for bbox in pred:
            if bbox['image_id'] == id and bbox['score'] > threshold: 
                x,y,w,h = bbox['bbox']
                pred_mask[int(y):int(y+h),int(x):int(x+w)] = 1 
        tmp_mask = gt_mask + pred_mask
        iou = len(np.where(tmp_mask==2)[0])/len(np.where(tmp_mask>=1)[0])
        dice = 2*len(np.where(tmp_mask==2)[0])/(len(np.where(gt_mask==1)[0])+len(np.where(pred_mask==1)[0]))
        tp_iou.append(iou)      
        tp_dice.append(dice)
        # if max(PTX_area) > 0 and max(PTX_area) <= 415*500:
        if gt['images'][id]['file_name'].replace('.png','') in size_LUT['s']:
            tp_iou_s.append(iou)
            tp_dice_s.append(dice)
            performance_detail.append({'iou': iou, 'dice': dice, 'size': 's'})

        elif gt['images'][id]['file_name'].replace('.png','') in size_LUT['m']:
            tp_iou_m.append(iou)
            tp_dice_m.append(dice)
            performance_detail.append({'iou': iou, 'dice': dice, 'size': 'm'})
        elif gt['images'][id]['file_name'].replace('.png','') in size_LUT['l']:
            tp_iou_l.append(iou)
            tp_dice_l.append(dice)
            performance_detail.append({'iou': iou, 'dice': dice, 'size': 'l'})
        else:
            print(id)
            raise

    print(len(performance_detail))
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
    
    # plt.hist(tp_dice_l_bootsrapping)
    # plt.savefig('hist.png') 
    
    tmp = {'all': tp_dice_bootsrapping, 's': remove_nan(tp_dice_s_bootsrapping), 'm': remove_nan(tp_dice_m_bootsrapping), 'l': remove_nan(tp_dice_l_bootsrapping)}
    
    with open('./prediction/dice_det_based_cad.json', 'w') as file:
        json.dump(tmp, file)
        

