import json

if __name__ == '__main__':
    gt = json.load(open('./prediction/NTUH_20_gt.json'))
    pred = json.load(open('./prediction/ensemble.json'))
    pred_2ls_negative_pid = json.load(open('./prediction/2cls_neg_list.json'))

    negative_image_id = []
    for image in gt['images']:
        if image['file_name'][:-4] in pred_2ls_negative_pid:
            negative_image_id.append(image['id'])
    
    pred_2cls = []
    # for bbox in pred:
    #     if bbox['image_id'] not in negative_image_id:
    #         pred_2cls.append(bbox)
    for bbox in pred:
        if bbox['image_id'] in negative_image_id:
            bbox['score'] = bbox['score']*0.2
        pred_2cls.append(bbox)
    print(len(pred))
    print(len(pred_2cls))


    with open('./prediction/ensemble_2cls.json', 'w') as file:
        json.dump(pred_2cls, file)
        