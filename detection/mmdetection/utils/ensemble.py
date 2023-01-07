import argparse
import json
from tqdm import tqdm
import numpy as np
import math

from ensemble_boxes import *

def json2dict(path):
    with open(path, mode='r') as file:
        data = json.load(file)
    return data

def ensemble(gt_path, prediction_paths, method, param=None):
    """
    Implementation of detection ensembling.
    Provided algorithms: NMS, soft-NMS, NMW, WBF

    Refence:
    [1] https://github.com/ZFTurbo/Weighted-Boxes-Fusion
    """
    if not param:
        param = {'iou_thr' : 0.5, 'skip_box_thr' : 0.0001, 'sigma': 0.1, 'weights': [1/len(prediction_paths)]*len(prediction_paths)}
    print(param)
    gt = json2dict(gt_path)

    model_output_list = []
    print(f"loading {len(prediction_paths)} models...")
    for i in range(len(prediction_paths)):
        model_output_list.append(json2dict(prediction_paths[i]))
    
    # Since the 'ensemble_boxes' requires the cooredinates been normalized, we need the image size
    id_size_LUT = {}
    for i in range(len(gt['images'])):
        id_size_LUT[gt['images'][i]['id']] = [gt['images'][i]['width'], gt['images'][i]['height']]

    output_ensemble = []
    for id in tqdm(id_size_LUT.keys()):
        boxes_list = []
        scores_list = []
        labels_list = []
    
        for output in model_output_list:
            boxes = []
            scores = []
            labels = []
            for box in output:
                if box['image_id'] != id:
                    continue
               
                xywh = box['bbox']
                xyxy = [xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]]
                xyxy_norm = [xyxy[0]/id_size_LUT[id][0], xyxy[1]/id_size_LUT[id][1], xyxy[2]/id_size_LUT[id][0], xyxy[3]/id_size_LUT[id][1]]
                boxes.append(xyxy_norm)
                scores.append(box['score'])
                labels.append(box['category_id'])
            boxes = np.array(boxes)
            boxes[boxes>1]=1
            boxes[boxes<0]=0
            boxes = boxes.tolist()
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
        boxes_list = [boxes for boxes in boxes_list if boxes != []]
        scores_list = [scores for scores in scores_list if scores != []]
        labels_list = [labels for labels in labels_list if labels != []]
        if method == 'nms':
            boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=param['weights'], iou_thr=param['iou_thr'])
        elif method == 'soft_nms':
            boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=param['weights'], iou_thr=param['iou_thr'])
        elif method == 'nmw':
            boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=param['weights'], iou_thr=param['iou_thr'])
        elif method == 'wbf':
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=param['weights'], iou_thr=param['iou_thr'])
        else:
            raise NameError(f'the algorithm name {method} is not supported')
        
        
        for xyxy_norm, score, label in zip(boxes, scores, labels):
            xyxy = [xyxy_norm[0]*id_size_LUT[id][0], xyxy_norm[1]*id_size_LUT[id][1], xyxy_norm[2]*id_size_LUT[id][0], xyxy_norm[3]*id_size_LUT[id][1]]
            xywh = [xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]]
            xywh = [coordinate.tolist() for coordinate in xywh]
            if math.isnan(xywh[0]) or math.isnan(xywh[1]) or math.isnan(xywh[2]) or math.isnan(xywh[3]):
                continue
            output_ensemble.append({"image_id": int(id), "bbox": xywh, "score": float(score), "category_id": int(label)})
    return output_ensemble

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', help='ground truth path')
    parser.add_argument('--predictions', nargs='+', type=str, help='jsons of predicitons, could be multiple.')
    parser.add_argument('--method', default='wbf', help='ensemble algorithm to fuse bboxs')
    parser.add_argument('--output-name', default='output.json', help='output file name')
    opt = parser.parse_args()

    
    method = 'wbf'
    iou_thr = 0.60445
    skip_box_thr = 0.0001
    sigma = 0.1
    weights = [
        0.72412,
        0.20912,
        0.20482,
        0.41983,
        0.33407
    ]
    param = {'iou_thr' : iou_thr, 'skip_box_thr' : skip_box_thr, 'sigma': sigma, 'weights': weights}
    
    output_ensemble = ensemble(opt.gt, opt.predictions, opt.method, opt.param)
    
    with open(opt.output_name, mode='w') as file:
        json.dump(output_ensemble, file)
