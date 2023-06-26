import argparse
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, auc, roc_auc_score, average_precision_score
import random


def set_random_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed) 
def read_json(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

def thresholding(prob, threshold):
    output = [0] * len(prob)
    for i in range(len(prob)):
        if prob[i] > threshold:
            output[i] = 1
        else:
            output[i] = 0
    return output


def statistic(data):
    print(np.mean(data),',' , np.std(data, ddof=1), (np.percentile(data, 2.5), np.percentile(data, 97.5)))


def sensivity_specifity_cutoff(y_true, y_score):
    '''Find data-driven cut-off for classification
    
    Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.
    
    Parameters
    ----------
    
    y_true : array, shape = [n_samples]
        True binary labels.
        
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).
        
    References
    ----------
    
    Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
    Journal of clinical epidemiology, 59(8), 798-801.
    
    Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
    prediction models and markers: evaluation of predictions and classifications.
    Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.
    
    Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
    of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox-size', type=str, default='overall', help='overall, s, m, l')
    opt = parser.parse_args()
    gt_dict = read_json('../../data/10f_traininglist/holdout.json')
    prediction_dict = read_json('./prediction/ensemble_NTUH_1519.json')
    bbox_size = opt.bbox_size

    gt_class = [0] * len(gt_dict['images'])
    
    for bbox in gt_dict['annotations']:
        gt_class[bbox["image_id"]] = 1
    pred_class = [0] * len(gt_dict['images'])
    pid = []
    for img_id in range(len(gt_dict['images'])):
        bbox_conf = []
        pid.append(gt_dict['images'][img_id]['file_name'][:-4])
        for bbox in prediction_dict:
            if bbox['image_id'] == img_id:
                bbox_conf.append(bbox['score'])
        if len(bbox_conf) != 0:
            pred_class[img_id] = max(bbox_conf)
    threshold = sensivity_specifity_cutoff(gt_class, pred_class)
    # print(threshold)



    gt_dict = read_json('./prediction/NTUH_20_gt.json')
    prediction_dict = read_json('./prediction/ensemble_2cls.json')
    bbox_size = opt.bbox_size

    gt_class = [0] * len(gt_dict['images'])
    
    for bbox in gt_dict['annotations']:
        gt_class[bbox["image_id"]] = 1
    pred_class = [0] * len(gt_dict['images'])
    pid = []
    for img_id in range(len(gt_dict['images'])):
        bbox_conf = []
        pid.append(gt_dict['images'][img_id]['file_name'][:-4])
        for bbox in prediction_dict:
            if bbox['image_id'] == img_id:
                bbox_conf.append(bbox['score'])
        if len(bbox_conf) != 0:
            pred_class[img_id] = max(bbox_conf)
    
    # size-wise
    tmp_gt = []
    tmp_pred = []
    with open('./prediction/NTUH_20_size_LUT.json', 'r') as file:
        size_LUT = json.load(file)
    
    negative = list(set(pid)-set(size_LUT['s'])-set(size_LUT['m'])-set(size_LUT['l']))
    for i in range(len(gt_class)):
        if  bbox_size == 'overall' or (pid[i] in negative or pid[i] in size_LUT[bbox_size]):
            tmp_gt.append(gt_class[i])
            tmp_pred.append(pred_class[i])
    gt_class = tmp_gt
    pred_class = tmp_pred
    # print(len(size_LUT['s']))
    # print(len(size_LUT['m']))
    # print(len(size_LUT['l']))

    AUC_bootstrap = []
    AP_bootstrap = []
    specificity_bootstrap = []
    sensitivity_bootstrap = []
    NPV_bootstrap = []
    PPV_bootstrap = []
    iteration=1000
    set_random_seed()
    for i in tqdm(range(iteration)):
        tmp_gt_class = []
        tmp_pred_class = []
        for id in np.random.randint(len(gt_class), size=len(gt_class)):
    
            tmp_gt_class.append(gt_class[id])
            tmp_pred_class.append(pred_class[id])

        y = np.array(tmp_gt_class)
        pred = np.array(tmp_pred_class)
        fpr, tpr, thresholds = roc_curve(y, pred)
        # print(f'AUROC: {auc(fpr, tpr)}')
        AUC = roc_auc_score(y, pred)
        AP = average_precision_score(y, pred)
        
        # threshold = 0.20566484218048947
 
        tmp_pred_class = thresholding(tmp_pred_class, threshold)
        
        tn, fp, fn, tp = confusion_matrix(np.array(tmp_gt_class), np.array(tmp_pred_class)).ravel()
        p = tp / (tp + fp)

        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        NPV = tn / (tn + fn)
        PPV = tp / (tp + fp)
        AUC_bootstrap.append(AUC)
        AP_bootstrap.append(AP)
        specificity_bootstrap.append(specificity)
        sensitivity_bootstrap.append(sensitivity)
        NPV_bootstrap.append(NPV)
        PPV_bootstrap.append(PPV)
    print("Detection-based CAD system:")
    print("AUC")
    statistic(AUC_bootstrap)
    print("AP")
    statistic(AP_bootstrap)
    print("Sensitivity")
    statistic(sensitivity_bootstrap)
    print("Specificity")
    statistic(specificity_bootstrap)
    print("PPV")
    statistic(PPV_bootstrap)
    print("NPV")
    statistic(NPV_bootstrap)