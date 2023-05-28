import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate(gt, prediction):
    cocoGt=COCO(gt)

    cocoDt=cocoGt.loadRes(prediction)

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
    parser.add_argument('--prediction', help='prediction json')
    opt = parser.parse_args()
    
    metric = evaluate(opt.gt, opt.prediction)
    print(metric)


    