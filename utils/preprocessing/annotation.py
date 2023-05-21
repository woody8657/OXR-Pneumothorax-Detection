import os
import json
import numpy as np
import cv2
import skimage.measure

class Annotation():
    def __init__(self, json_path):
        self.json = json.load(open(json_path))
        self.accessionNumber = self.json['accessionNumber']
        self.shape = self.json['size'][::-1]

    def get_bbox(self):
        bbox_list = []
        for j in range(len(self.json['shapes'])):
            if 'tagAnno' in self.json['shapes'][j] and 'type' in self.json['shapes'][j]:
                if self.json['shapes'][j]['tagAnno'] == "Pneumothorax" or self.json['shapes'][j]['tagAnno'] == "Pneumothorax " and self.json['shapes'][j]['type'] == "Rectangle":
                    x_a, y_a = self.json['shapes'][j]['a']
                    x_b, y_b = self.json['shapes'][j]['b']
                    # left up corner
                    (x, y, w, h) = map(int,(round(min(x_a, x_b)), round(min(y_a, y_b)), round(max(x_a, x_b))-round(min(x_a, x_b)), round(max(y_a, y_b)-min(y_a, y_b))))
                    x, y, w, h = round(min(x_a, x_b)), round(min(y_a, y_b)), round(max(x_a, x_b))-round(min(x_a, x_b)), round(max(y_a, y_b)-min(y_a, y_b))
                    bbox_list.append([x,y,w,h])
            else:
                pass
            
        
        return bbox_list

    def get_plotted_img(self, img, bbox):
        # img = self.get_img()

        for x,y,w,h in bbox:
            img = cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 5)
            # score = np.random.rand() / 10 + 0.9
            # cv2.putText(img, "Pneumothorax {:.2f}".format(score), (int(x), int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        return img
    

    def grid_bbox2mask(self):
        mask = np.zeros((int(self.json['size'][1]), int(self.json['size'][0])))
        bbox_list = []
        for j in range(len(self.json['shapes'])):
            if 'tagAnno' in self.json['shapes'][j] and 'type' in self.json['shapes'][j]:
                if self.json['shapes'][j]['tagAnno'] == "Pneumothorax" or self.json['shapes'][j]['tagAnno'] == "Pneumothorax " and self.json['shapes'][j]['type'] == "Rectangle":
                    x_a, y_a = self.json['shapes'][j]['a']
                    x_b, y_b = self.json['shapes'][j]['b']
                    x, y, w, h = int(min(x_a, x_b)), int(min(y_a, y_b)), int(max(x_a, x_b)-min(x_a, x_b)), int(max(y_a, y_b)-min(y_a, y_b))        
                    mask[y:y+h,x:x+w] = 255
                    bbox_list.append([x,y,w,h])
        
        return mask, bbox_list

    def mask2bbox(self, mask):
        mask[mask<125]=0
        mask[mask>=125]=1
        tmp_mask = np.zeros((mask.shape[0], mask.shape[1]))
        mask = skimage.measure.label(mask,connectivity=1)
        xyxy_bbox = np.empty((0,4), int)
        
        for count, region in enumerate(skimage.measure.regionprops(mask)):
            minr, minc, maxr, maxc = region.bbox
            xyxy_bbox = np.append(xyxy_bbox, np.expand_dims(np.array([minc, minr, maxc, maxr]), axis=0), axis=0)
            # if count == 0:
            tmp_mask[minr:maxr, minc:maxc] = 255
            # if count == 1:
            #     tmp_mask[minr:maxr, minc:maxc,:] = np.array([0,0,120])

        xywh_bbox = []
        for i in range(xyxy_bbox.shape[0]):
            (x1,y1) = map(int,(xyxy_bbox[i,0], xyxy_bbox[i,1]))
            (x2,y2) = map(int,(xyxy_bbox[i,2], xyxy_bbox[i,3]))
            xywh_bbox.append([x1, y1, x2-x1, y2-y1])
            
        return xywh_bbox, tmp_mask



if __name__ == "__main__":
    # 04/11 searching initial annotation
    import glob
    json_list = glob.glob("/home/u/woody8657/data/C426_Original/C426_01/Annotations/**/*.json")

    import pandas as pd
    def acc2pid(acc):
        df = pd.read_csv('/home/u/woody8657/tmp/Annotation_cleaning/G1_01_chest_tube.csv')
        try:
            idx = df[df['ACCESSNO2']==acc].index.values[0]
        except:
            return 0
        return df.iloc[idx,0]

    data = []
    for json_path in json_list:
        annotation = Annotation(json_path)
        try:
            pid = acc2pid(annotation.json['accessionNumber'])
            bbox_list = annotation.get_bbox()
            if bbox_list == 0 or pid == 0 or bbox_list == []:
                raise
            # if len(os.listdir(os.path.join("/home/u/woody8657/tmp/Annotation_cleaning/annotation_pid", pid))) == 2:
            # print(len(os.listdir(os.path.join("/home/u/woody8657/tmp/Annotation_cleaning/annotation_pid", pid))))
            data.append(pid)
        except:
            pass
    print(len(data))

    data2 = []
    for pid in os.listdir('/home/u/woody8657/tmp/Annotation_cleaning/annotation_pid'):
        if len(os.listdir(os.path.join('/home/u/woody8657/tmp/Annotation_cleaning/annotation_pid', pid))) ==2:
            data2.append(data2)

    print(len(data2))
    # print(set(data1).intersection(set(data2)))
    # raise

    dice_list1 = []
    iou_list1 = []
    import os
    pid_included = ['P214260000386', 'P214260000226', 'P214260000299', 'P214260000646', 'P214260002194', 'P214260000597', 'P214260000080', 'P214260000023', 'P214260001176', 'P214260001154', 'P214260001053', 'P214260001525']
    for json_path in json_list:
        annotation = Annotation(json_path)
        # try:
        pid = acc2pid(annotation.json['accessionNumber'])
        if pid != 0 and pid in pid_included:
            annotation1 = Annotation(json_path)
            # print(annotation1.json)
            annotation2 = Annotation(os.path.join('/home/u/woody8657/tmp/Annotation_cleaning/annotation_pid',pid,'0.json'))
            mask1, _ = annotation1.grid_bbox2mask()
            mask2, _ = annotation2.grid_bbox2mask()
            # print(mask2)
            _, mask2 = annotation2.mask2bbox(mask2)
            mask_tmp = mask1 + mask2
            iou = len(np.argwhere(mask_tmp >= 510))/len(np.argwhere(mask_tmp >= 255))
            dice = 2*len(np.argwhere(mask_tmp >= 510))/(len(np.argwhere(mask_tmp == 255))+2*len(np.argwhere(mask_tmp == 510)))
            iou_list1.append(iou+0.00001)
            dice_list1.append(dice+0.00001)
        # except:
        #     print(pid)
        #     continue


    print(iou_list1)
    dice_list2 = []
    iou_list2 = []
    import os
    pid_included = ['P214260000386', 'P214260000226', 'P214260000299', 'P214260000646', 'P214260002194', 'P214260000597', 'P214260000080', 'P214260000023', 'P214260001176', 'P214260001154', 'P214260001053', 'P214260001525']
    for pid in pid_included:
        
        annotation1 = Annotation(os.path.join('/home/u/woody8657/tmp/Annotation_cleaning/annotation_pid',pid,'1.json'))
        annotation2 = Annotation(os.path.join('/home/u/woody8657/tmp/Annotation_cleaning/annotation_pid',pid,'0.json'))
        mask1, _ = annotation1.grid_bbox2mask()
        _, mask1 = annotation2.mask2bbox(mask1)
        mask2, _ = annotation2.grid_bbox2mask()
        _, mask2 = annotation2.mask2bbox(mask2)
        mask_tmp = mask1 + mask2
        iou = len(np.argwhere(mask_tmp >= 510))/len(np.argwhere(mask_tmp >= 255))
        iou_list2.append(iou)
        dice = 2*len(np.argwhere(mask_tmp >= 510))/(len(np.argwhere(mask_tmp == 255))+2*len(np.argwhere(mask_tmp == 510)))
        dice_list2.append(dice+0.00001)
    
    print(dice_list2)
    iteration = 1000
    def statistic(data):
        print(np.mean(data),',' , np.std(data, ddof=1), (np.percentile(data, 2.5), np.percentile(data, 97.5)))
    stat1 = []
    stat2 = []
    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)  
    for _ in range(iteration):
        tmp1 = []
        for id in np.random.randint(len(dice_list1), size=len(dice_list1)):
            tmp1.append(dice_list1[id])
        stat1.append(sum(tmp1)/len(tmp1))
        tmp2 = []
        for id in np.random.randint(len(dice_list2), size=len(dice_list2)):
            tmp2.append(dice_list2[id])
        stat2.append(sum(tmp2)/len(tmp2))
    
    statistic(stat1)
    statistic(stat2)
    from scipy import stats
    a, b = stats.ttest_rel(np.array(stat2).astype('float64'), np.array(stat1).astype('float64'), axis=0, nan_policy='propagate', alternative='greater')        
    print(b)

    import matplotlib.pyplot as plt
    data = [np.array(dice_list1), np.array(dice_list2)]
    
    # fig = plt.figure(figsize =(10, 7))
    
    fig7, ax7 = plt.subplots()
    ax7.set_title("")
    ax7.boxplot(data)
    # Creating axes instance
    # ax = fig.add_axes([0, 0, 1, 1])
    
    # Creating plot
    # bp = ax.boxplot(data)
    ax7.set_xticklabels(['without grid limitation', 'grid annotation'])
 
    # Adding title
    # plt.title("Box plot of consistency measured by IoU")
    # show plot
    plt.savefig('box_dice.png')
    

                
    raise

    # annotations2 = []
    # for key, value in stat.items():
    #     if value >= 2 :
    #         annotations2.append(key)

    # import os
    # import json
    # for id, uid in enumerate(annotations2):
    #     save_dir = f'./before_grid/{id}'
    #     os.makedirs(save_dir, exist_ok=True)
    #     count = 0
    #     for json_path in json_list:
    #         if uid in json_path:
    #             annotation = Annotation(json_path)
    #             # print(count)
    #             bbox = annotation.get_bbox()
    #             with open(os.path.join(save_dir, f"{count}.json"), 'w') as f:
    #                 json.dump(annotation.json, f)
    #             count = count + 1
    import os
    analysis_dir = '/home/u/woody8657/projs/OXR-Pneumothorax-Detection/utils/preprocessing/before_grid'
    # analysis_dir = '/home/u/woody8657/tmp/Annotation_cleaning/annotation_pid'
    mask_list_overall = []
    area = []
    from tqdm import tqdm
    for id in tqdm(os.listdir(analysis_dir)):
        if len(os.listdir(os.path.join(analysis_dir, id))) == 2:
            mask_list = []
            flag = False
            for filename in os.listdir(os.path.join(analysis_dir, id)):
                annotation = Annotation(os.path.join(analysis_dir, id, filename))

                bbox = annotation.get_bbox()

                if bbox == 0:
                    mask_list.append(0)
                    flag = True
                else:
                    mask, _ = annotation.grid_bbox2mask()
                    mask_list.append(mask)
            if not flag:
                mask_list_overall.append(mask_list)
    iou_list1 = []
    for mask1, mask2 in tqdm(mask_list_overall):
        try:
            mask_tmp = mask1 + mask2
            iou = len(np.argwhere(mask_tmp >= 510))/len(np.argwhere(mask_tmp >= 255))
            iou_list1.append(iou)
        except:
            area.append(max(len(np.argwhere(mask1 >= 255)), len(np.argwhere(mask2 >= 255))))
            
            mask1 = cv2.resize(mask1, (mask2.shape[1], mask2.shape[0]))
            mask_tmp = mask1 + mask2
            iou = len(np.argwhere(mask_tmp >= 510))/len(np.argwhere(mask_tmp >= 255))
            iou_list1.append(iou)
            pass
    print(sum(iou_list1)/len(iou_list1))
    import os
    # analysis_dir = '/home/u/woody8657/projs/OXR-Pneumothorax-Detection/utils/preprocessing/before_grid'
    analysis_dir = '/home/u/woody8657/tmp/Annotation_cleaning/annotation_pid'
    mask_list_overall = []
    from tqdm import tqdm
    for id in tqdm(os.listdir(analysis_dir)):
        if len(os.listdir(os.path.join(analysis_dir, id))) == 2:
            mask_list = []
            flag = False
            for filename in os.listdir(os.path.join(analysis_dir, id)):
                annotation = Annotation(os.path.join(analysis_dir, id, filename))

                bbox = annotation.get_bbox()

                if bbox == 0:
                    mask_list.append(0)
                    flag = True
                else:
                    mask, _ = annotation.grid_bbox2mask()
                    _, mask = annotation.mask2bbox(mask)
                    mask_list.append(mask)
            if not flag:
                mask_list_overall.append(mask_list)
    iou_list2 = []
    count = 0
    for mask1, mask2 in tqdm(mask_list_overall):
        if max(len(np.argwhere(mask1 >= 255)), len(np.argwhere(mask2 >= 255))) > 101871:
            mask_tmp = mask1 + mask2
            iou = len(np.argwhere(mask_tmp >= 510))/len(np.argwhere(mask_tmp >= 255))
            iou_list2.append(iou)
            count = count+1
            if count == 98:
                break

      

    print(sum(iou_list2)/len(iou_list2))



    import matplotlib.pyplot as plt
    import numpy as np
    
    
    # Creating dataset
    
    data = [np.array(iou_list1), np.array(iou_list2)]
    
    fig = plt.figure(figsize =(10, 7))
    
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Creating plot
    bp = ax.boxplot(data)
    
    # show plot
    plt.savefig('box.png')
                
