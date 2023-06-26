import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np
import copy
from tqdm import tqdm
import scipy.stats as st

def evaluate(gt, json):
    cocoGt=COCO(gt)

    cocoDt=cocoGt.loadRes(json)

    imgIds=sorted(cocoGt.getImgIds())
    # imgId = imgIds

    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    
    cocoEval.params.imgIds  = imgIds
    
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats

def read_json(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

def statistic(data):
    print(np.mean(data), np.std(data), st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) )
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', help='ground truth json')
    parser.add_argument('--pred', help='prediction json')
    opt = parser.parse_args()

    gt_dict = read_json(opt.gt)
    prediction_dict = read_json(opt.pred)

    iteration = 1000
    result = []
    for _ in tqdm(range(iteration)):
        gt_tmp_dict = {}
        bootstrap_imgs = []
        bootstrap_bboxes_gt = []
        bootstrap_bboxes_prediction = []

        img_count = 0
        bbox_count_gt = 0
        for sample in range(len(gt_dict['images'])): 
            img_id = np.random.randint(len(gt_dict['images'])) 
            img_copy = copy.deepcopy(gt_dict['images'][img_id])
            
            img_copy['id'] = img_count
            img_copy['file_name'] = str(img_count) + '.png'
            bootstrap_imgs.append(img_copy)
            
            
            for bbox in gt_dict['annotations']:
                if bbox['image_id'] == img_id:
                    bbox_copy = copy.deepcopy(bbox)
                    bbox_copy['id'] = bbox_count_gt
                    bbox_copy['image_id'] = img_count
                    bootstrap_bboxes_gt.append(bbox_copy)
                    bbox_count_gt = bbox_count_gt + 1

            for bbox in prediction_dict:
                if bbox['image_id'] == img_id:
                    bbox_copy = copy.deepcopy(bbox)
                    bbox_copy['image_id'] = img_count
                    bootstrap_bboxes_prediction.append(bbox_copy)

            img_count = img_count + 1
        gt_tmp_dict = {'images': bootstrap_imgs, 'annotations': bootstrap_bboxes_gt, 'categories': gt_dict['categories']}
        
        prediction_tmp_dict = bootstrap_bboxes_prediction
        with open('./prediction/tmp_gt.json', 'w') as file:
            json.dump(gt_tmp_dict, file)
        with open('./prediction/tmp_prediction.json', 'w') as file:
            json.dump(prediction_tmp_dict, file)
        

        metric = evaluate('./prediction/tmp_gt.json', './prediction/tmp_prediction.json')
        result.append(list(metric))
    statistic(np.array(result)[:,1])
    print(np.array(result).shape)
    with open('./prediction/det_based_cad_bootstrap.json', 'w') as file:
        json.dump(result, file)
    


    