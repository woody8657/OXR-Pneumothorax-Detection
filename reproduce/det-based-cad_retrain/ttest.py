# two sample student t test
import numpy as np
from scipy import stats
import json
import numpy as np

def p_vaue(mean1, std1, mean2, std2, nobs1=10000, nobs2=10000):
    modified_std1 = np.sqrt(np.float32(nobs1)/np.float32(nobs1-1)) * std1
    modified_std2 = np.sqrt(np.float32(nobs2)/np.float32(nobs2-1)) * std2
    (statistic, pvalue) = stats.ttest_ind_from_stats(mean1=mean1, std1=modified_std1, nobs1=10, mean2=mean2, std2=modified_std2, nobs2=10)
    return pvalue

if __name__ == '__main__':
    
    with open('./prediction/dice_det_based_cad.json', 'r') as file:
        pred = json.load(file)
    with open('./prediction/dice_annotator.json', 'r') as file:
        annotator = json.load(file)
    a, b = stats.ttest_rel(np.array(pred['all']).astype('float64'), np.array(annotator['all']).astype('float64'), axis=0, nan_policy='propagate', alternative='greater')        
    print(b)
    a, b = stats.ttest_rel(np.array(pred['l']).astype('float64'), np.array(annotator['l']).astype('float64'), axis=0, nan_policy='propagate', alternative='greater')        
    print(b)
    a, b = stats.ttest_rel(np.array(pred['m']).astype('float64'), np.array(annotator['m']).astype('float64'), axis=0, nan_policy='propagate', alternative='greater')        
    print(b)
    a, b = stats.ttest_rel(np.array(pred['s']).astype('float64'), np.array(annotator['s']).astype('float64'), axis=0, nan_policy='propagate', alternative='greater')        
    print(b)
