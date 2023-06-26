import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate(gt, json):
    cocoGt=COCO(gt)

    cocoDt=cocoGt.loadRes(json)

    imgIds=sorted(cocoGt.getImgIds())

    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', help='ground truth json')
    parser.add_argument('--pred', help='prediction json')
    opt = parser.parse_args()
    
    metric = evaluate(opt.gt, opt.pred)
    print(metric)


    