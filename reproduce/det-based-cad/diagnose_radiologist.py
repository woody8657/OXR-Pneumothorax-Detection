import argparse
import os
import pandas as pd
import json
import numpy as np
import random
from sklearn.metrics import roc_auc_score,roc_curve,auc,precision_recall_curve,classification_report, confusion_matrix, plot_confusion_matrix, average_precision_score
from tqdm import tqdm
import scipy.stats as st

def remove_nan(input):
    output = []
    for i in input:
        if i > 0:
            output.append(i)
        else:
            output.append(0)
    return output

def set_random_seed():
    seed = 57
    random.seed(seed)
    np.random.seed(seed)  
def statistic(data):
    print(np.mean(data),',' , np.std(data, ddof=1), (np.percentile(data, 2.5), np.percentile(data, 97.5)))

def performance_per_size(size_json_path, report_csv_path , size=None):
    report = pd.read_csv(report_csv_path)
    with open(size_json_path, 'r') as file:
        pid_size_dict = json.load(file)
    no_chest_tube = []
    for i in range(report.shape[0]):
        if report.iloc[i,4] != 1:
            no_chest_tube.append([report.iloc[i,1], report.iloc[i,3], report.iloc[i,-4]])
    include_patient = []
    gt = []
    pred = []
    if size == "overall":
        for single_patieent in no_chest_tube:
            include_patient.append(single_patieent)
            gt.append(single_patieent[1])
            pred.append(single_patieent[2])
    elif size in ['s', 'm', 'l']:
        print(len(pid_size_dict[size]))
        for single_patieent in no_chest_tube:
            if single_patieent[1] == 0 or single_patieent[0] in pid_size_dict[size]:
                include_patient.append(single_patieent)
                gt.append(single_patieent[1])
                pred.append(single_patieent[2])
    else:
        print(size)
        raise
    # bootstrapping
    set_random_seed()
    iteration = 1000
    specificity = []
    sensitivity = []
    NPV = []
    PPV = []
    AP = []
    AUC = []
    for i in tqdm(range(iteration)):
        index = np.random.randint(len(gt), size=len(gt))
        tmp_gt = []
        tmp_pred = []
        for j in range(len(gt)):
            tmp_gt.append(gt[index[j]])
            tmp_pred.append(pred[index[j]])

    
        tn, fp, fn, tp = confusion_matrix(tmp_gt, tmp_pred).ravel()
        specificity.append(tn / (tn + fp))
        sensitivity.append(tp / (tp + fn))
        NPV.append(tn / (tn + fn))
        PPV.append(tp / (tp + fp))
        AUC.append(roc_auc_score(tmp_gt, tmp_pred))
        AP.append(average_precision_score(tmp_gt, tmp_pred))
    # print(size)
    print("Radiologist report dignosis")
    print("AUC")
    statistic(remove_nan(AUC))
    print("AP")
    statistic(remove_nan(AP))
    print("Sensitivity")
    statistic(remove_nan(sensitivity))
    print("Specificity")
    statistic(remove_nan(specificity))
    print("PPV")
    statistic(remove_nan(PPV))
    print("NPV")
    statistic(remove_nan(NPV))
    
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox-size', type=str, default='overall', help='overall, s, m, l')
    opt = parser.parse_args()

    report_csv_path = './prediction/NTUH_20_report.csv'
    size_json_path = './prediction/NTUH_20_size_LUT.json'
    performance_per_size(size_json_path, report_csv_path, size=opt.bbox_size)
    

        
        