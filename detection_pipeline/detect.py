import cv2
import os
import argparse

from .detector_and_classifier import GreenhouseModel

# To use from terminal:
# python3 detect.py --imgsz 2016 --conf_thres 0.4 --iou_thres 0.1 --max_det 150 --output_fname output.jpg


def run(source=None,
        imgsz=(2016, 2016),
        conf_thres=0.1,
        iou_thres=0.4,
        max_det=150,
        save=False,
        output_fname='output.jpg'
        ):
    model = GreenhouseModel(imgsz=imgsz, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
    prediction = model.predict(source)
    if save:
        cv2.imwrite(os.path.join(os.getcwd(), output_fname), prediction)
        print(f'Prediction saved as {os.path.join(os.getcwd(), output_fname)}')
    return prediction

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='test_image.jpg')
    parser.add_argument('--imgsz', type=int, default=2016)
    parser.add_argument('--conf_thres', type=float, default=0.1)
    parser.add_argument('--iou_thres', type=float, default=0.4)
    parser.add_argument('--max_det', type=int, default=150)
    parser.add_argument('--output_fname', type=str, default='output.jpg')
    opt = parser.parse_args()
    opt.imgsz = [opt.imgsz]
    opt.imgsz *= 2
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
