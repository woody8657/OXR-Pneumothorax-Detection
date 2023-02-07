import os
import argparse
import json
import cv2

def get_plotted_img(img, bbox_list, include_score=False):
        if include_score:
            for x,y,w,h,score in bbox_list:
                img = cv2.rectangle(img, (int(x),int(y)), (int(x+w), int(y+h)), (0, 0, 255), 5)
                cv2.putText(img, "Pneumothorax {:.2f}".format(score), (int(x), int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            for x,y,w,h in bbox_list:
                img = cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 5)
        return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref-json', help="only needed when bbox-json doesn't include image info, e.g, inference output.")
    parser.add_argument('--bbox-json', help='inference result')
    parser.add_argument('--img-prefix', help='img direcory')
    parser.add_argument('--query-imgs', nargs='+', type=str, help='images file name used to visulized')
    parser.add_argument('--conf-threshold', default=0.5, type=float, help='confidence threhold to display bounding box')
    parser.add_argument('--save-dir', default='inference_images', type=str, help='images used to visulized')
    opt = parser.parse_args()

    os.makedirs(opt.save_dir, exist_ok=True)

    with open(opt.ref_json, 'r') as f:
        ref_dict = json.load(f)
    with open(opt.bbox_json, 'r') as f:
        annotations = json.load(f)

    img_dict = {}
    for file_name in opt.query_imgs:
        img = cv2.imread(os.path.join(opt.img_prefix, file_name))

        bbox_list = []
        # search the image id to query the annotations 
        try:
            for image_info in annotations['images']:
                if image_info['file_name'] == file_name:
                    image_id = image_info['id']
                    break

            for bbox_info in annotations['annotations']:
                if bbox_info['image_id'] == image_id:
                    if 'score' in bbox_info.keys():
                        bbox_list.append(bbox_info['bbox']+bbox_info['score'])
                    else:
                        bbox_list.append(bbox_info['bbox'])
            
            img = get_plotted_img(img, bbox_list)

        except TypeError:
            for image_info in ref_dict['images']:
                if image_info['file_name'] == file_name:
                    image_id = image_info['id']
                    break

            for bbox_info in annotations:
                if bbox_info['image_id'] == image_id:
                    if 'score' in bbox_info.keys():
                        if bbox_info['score'] < opt.conf_threshold:
                            continue
                        bbox_list.append(bbox_info['bbox']+[bbox_info['score']])
                    else:
                        bbox_list.append(bbox_info['bbox'])
            
            img = get_plotted_img(img, bbox_list, include_score=True)

        cv2.imwrite(os.path.join(opt.save_dir, file_name), img)
        

        
        

