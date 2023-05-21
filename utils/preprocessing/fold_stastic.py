import os
import json
import pandas as pd




def list_intersect(list1, list2):
    return list(set(list1).intersection(list2))

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

    print(f"whole, PTX: {len(PTX)}, total: {len(PTX)+len(no_PTX)}, ratio: {len(PTX)/(len(PTX)+len(no_PTX))}")

    # split
    with open('split_10f_fixed.json', 'r') as file:
        split = json.load(file)
    
    for i in range(1,11):
        key = 'test_fold' + str(i)
        ptx = len(list_intersect(split[key], PTX))
        total = len(split[key])
        # print(f"fold:{i},PTX: {ptx}, total: {total}, ratio: {ptx/total}")
        for j in range(1,i):
            key_1 = 'test_fold' + str(j)
            print(f"intersection: {len(list_intersect(split[key],split['holdout']))}")

    ptx = len(list_intersect(split['holdout'], PTX))
    total = len(split['holdout'])
    print(f"holdout ,PTX: {ptx}, total: {total}, ratio: {ptx/total}")