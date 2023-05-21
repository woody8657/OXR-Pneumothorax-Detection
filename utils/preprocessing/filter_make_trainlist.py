import os
from matplotlib.font_manager import json_dump
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import json


def num_intersection(a,b):

    return len(set(a).intersection(set(b)))

if __name__ == '__main__':
    
    df = pd.read_csv('./report_no_pneumothorax.csv')

    idx_PTX = df[df['氣胸']==1].index.values
    idx_CT = df[df['胸管']==1].index.values
    no_PTX = list(set(df.iloc[:,0].tolist()) - set(df.iloc[idx_PTX,0].tolist()).union(set(df.iloc[idx_CT,0].tolist())))
   

    pid_list = os.listdir('/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images')
    pid_list = [file_name[:-4] for file_name in pid_list]

    PTX = []
    for pid in pid_list:
        if os.path.isfile(os.path.join('./annotation_pid_review', pid, 'label.json')):
            PTX.append(pid)

    # train, test = train_test_split(PTX+no_PTX, test_size=0.2, random_state=42)
    train = PTX + no_PTX

    print(len(train))
    
    
    print(num_intersection(train,PTX))
    print(num_intersection(train,no_PTX))

    

    train = np.array(train)
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    kf.get_n_splits(train)
    train_5fold = []
    test_5fold = []
    for train_index, test_index in kf.split(train):
        X_train, X_test = train[train_index], train[test_index]
        train_5fold.append(X_train.tolist())
        test_5fold.append(X_test.tolist())
        print(f'{num_intersection(X_train.tolist(),PTX)/len(X_train)}')
    print(test_5fold[0][:5])
    save_dict ={
        # 'holdout': test,
        'train_fold1':train_5fold[0], 'test_fold1':test_5fold[0],
        'train_fold2':train_5fold[1], 'test_fold2':test_5fold[1],
        'train_fold3':train_5fold[2], 'test_fold3':test_5fold[2],
        'train_fold4':train_5fold[3], 'test_fold4':test_5fold[3],
        'train_fold5':train_5fold[4], 'test_fold5':test_5fold[4],
        'train_fold6':train_5fold[5], 'test_fold6':test_5fold[5],
        'train_fold7':train_5fold[6], 'test_fold7':test_5fold[6],
        'train_fold8':train_5fold[7], 'test_fold8':test_5fold[7],
        'train_fold9':train_5fold[8], 'test_fold9':test_5fold[8],
        'train_fold10':train_5fold[9], 'test_fold10':test_5fold[9],
        }
    print('--------------')
    for i in range(10):
        # print(num_intersection(train_5fold[i], test_5fold[i]))
        print(len(test_5fold[i]), num_intersection(test_5fold[i], PTX), num_intersection(test_5fold[i], PTX)/len(test_5fold[i]))
    print(save_dict.keys())
    save_path = '10f_traininglist_noholdout'
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'split_10f_fixed.json'), 'w') as file:
        json.dump(save_dict, file)