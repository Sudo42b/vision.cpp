# Yolov9t Sequential model inference demo code

from utils import (non_max_suppression, 
                   letterbox, 
                   increment_path, 
                   yaml_load, 
                   smart_inference_mode, 
                   select_device,
                   scale_boxes, 
                   Annotator, 
                   colors, 
                   check_img_size)
from yolov9t import YOLOv9t_Seq
from torch import nn, device
from pathlib import Path
import numpy as np
import argparse
import torch
import cv2
import os

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov9t_converted.pth', help='model path')
    parser.add_argument('--source', type=str, default='./figure/cat-and-hat.jpg', help='image path')
    parser.add_argument('--data', type=str, default='./figure/coco.yaml', help='model yaml (for class names)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='h w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', default=False, action='store_true', help='use FP16')
    parser.add_argument('--line-thickness', default=3, type=int, help='bbox thickness (px)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    opt.imgsz = opt.imgsz * 2 if len(opt.imgsz) == 1 else opt.imgsz
    return opt


if __name__ == "__main__":
    ### 
    opt = parse_opt()
    
    source = opt.source
    half = opt.half

    device = select_device(opt.device)
    # model = DetectMultiBackend(opt.weights, device=device, data=opt.data, fp16=opt.half)
    
    model = YOLOv9t_Seq().cuda()
    model.load_state_dict(torch.load(opt.weights, map_location='cpu', weights_only=False))
    imgsz = check_img_size(opt.imgsz, s=model.stride)
    # Read image
    im0 = cv2.imread(source)  # BGR
    assert im0 is not None, f'Failed to load image: {source}'
    
    # Preprocess
    im = letterbox(im0, imgsz, stride=model.stride, auto=True)[0]
    model.eval()
    
    im = im.transpose((2, 0, 1))[::-1]  # HWC->CHW, BGR->RGB
    im = np.ascontiguousarray(im)
    names = yaml_load(opt.data)['names']
    im_tensor = torch.from_numpy(im).to(device)
    im_tensor = im_tensor.half() if half else im_tensor.float()
    im_tensor /= 255.0
    if im_tensor.ndimension() == 3:
        im_tensor = im_tensor.unsqueeze(0)
    print(f'Image shape: {im_tensor.shape}')  # torch.Size([1, 3, 640, 640])
    # Inference
    # model.warmup(imgsz=(1, 3, *imgsz))
    pred = model(im_tensor)
    # if not isinstance(pred, list):
    #     pred = [pred]
    def print_shape(p):
        if isinstance(p, (list, tuple)):
            print("is list")
            print(len(p))
            for p in p:
                print_shape(p)
        else:
            print("is not list")
            print(p.shape)
    
    # NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, max_det=opt.max_det)
    # Draw boxes
    annotator = Annotator(im0.copy(), line_width=opt.line_thickness, example=str(list(names.values())))
    
    det = pred[0]
    if len(det):
        det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in det:
            c = int(cls)
            cname = names[c] if c in names else str(c)
            label = None if opt.hide_labels else (cname if opt.hide_conf else f'{cname} {conf:.2f}')
            for i in xyxy:
                print(i.detach().cpu().numpy(), label)
            annotator.box_label(xyxy, label, color=colors(c, True))
    
    result = annotator.result()
    
    # Save dir
    save_dir = increment_path(Path('results'), exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    save_path = str((save_dir / Path(source).name).resolve())
    cv2.imwrite(save_path, result)
    print(f"Saved: {save_path}")