# -*- coding: utf-8 -*-
import argparse
#import torch.backends.cudnn as cudnn
from models.experimental import *
from utils.datasets import *
from utils.utils import *
import gc
import cv2
import scipy.ndimage as ndi
from skimage import measure,color
import datetime
import os
import numpy as np
from sklearn.cluster import MeanShift, KMeans
from sklearn.cluster import estimate_bandwidth
import matplotlib.pyplot as plt
import pylab
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd

def detect(save_img=False):    
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtmp') or source.startswith('http') or source.endswith('.txt')
    device = torch_utils.select_device(opt.device)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        #cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
        
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
        
    names = model.module.names if hasattr(model, 'module') else model.names
    print(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    
    for path, img, im0s, vid_cap in dataset:
        t1 = torch_utils.time_synchronized()
        #print(111,device)
        img = torch.from_numpy(img).to(device)
        img =  img.float()  # uint8 to fp16/32  img.half() if half else
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=False, agnostic=False)
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                 p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            camera = '31011811001180021011'
            now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            save_path = str(Path(out) / Path(p).name)
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                coords = []
                height = im0.shape[0]
                width = im0.shape[1]
                #region = []
                for *xyxy, conf, cls in det:

                    a1 = int((int(xyxy[0])+int(xyxy[2])) / 2)
                    a2 = int((int(xyxy[1])+int(xyxy[3])) / 2)
                    b = [a1, a2]
                    coords.append(b)
                    #region.append([(int(xyxy[2]) - int(xyxy[0]))*(int(xyxy[3]) - int(xyxy[1]))])
                    
                    gc.collect()
                X = np.mat(coords)
                #R = np.mat(region)
                bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                ms.fit(X)
                labels = ms.labels_
                cluster_centers = ms.cluster_centers_
                # 计算类别个数
                labels_unique = np.unique(labels)
                n_clusters = len(labels_unique)
                quantity = pd.Series(labels).value_counts()
                #画出聚类中心
                centers = cluster_centers
                res0Series = pd.Series(labels)
                dingd = []
                #aera_every_head = []
                for i in range(n_clusters):
                    res0 = res0Series[res0Series.values == i]
                    left = []
                    right = []
                    #every_head = []
                    for j in range(len(X[res0.index])):
                        left.append(float(X[res0.index][j][:, 0]))
                        right.append(float(X[res0.index][j][:, 1]))
                        #every_head.append(R[res0.index][j][:, 0])
                    #print(11111,every_head,sum(every_head))
                    
                    dingd.append([min(left),max(left),min(right),max(right)])
                    #aera_every_head.append(sum(every_head))
                #print(aera_every_head,len(aera_every_head))
                for c in range(len(centers[:, 0])):
                    c1 = (int(dingd[c][0]), int(dingd[c][2]))
                    c2 = (int(dingd[c][1]), int(dingd[c][3]))
                    tl = round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1
                    tf = max(tl - 1, 1)
                    left_x = (c2[0] - c1[0]) / 100
                    left_y = (c2[1] - c1[1]) / 100

                    mj = left_x * left_y
                    #print(222,area1)
                    area = float(quantity[c])

                    if mj > 0:                        
                        md = area / mj
                    else:
                        md = 0
                    cv2.rectangle(im0, (int(dingd[c][0]), int(dingd[c][2])), (int(dingd[c][1]),int(dingd[c][3])), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)
                    label_people = 'Count:{:s} | ratio:{:s}'.format(str(quantity[c]),str(np.round(md,2)))
                    t_size = cv2.getTextSize(label_people, 0, fontScale=tl / 3, thickness=tf)[0]

                    c2 = c1[0] + t_size[0] + 4, c1[1] + t_size[1] + 10
                    cv2.rectangle(im0, c1, c2, (255,0,255), -1, cv2.LINE_AA)
                    cv2.putText(im0,label_people,(c1[0] + 2, c1[1] + t_size[1] +2), 0, tl / 3,(255,255,255),thickness=tf, lineType=cv2.LINE_AA)
                
                
                cv2.imwrite(save_path+camera+'_'+str(now_time)+'.jpg', im0)

                print('****************')
                gc.collect()

            print(('%.3fs')%(t2 - t1))
           
        gc.collect()            
    gc.collect()                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/crowd/head_best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='I:/myproject/yolov5-master/111head/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='111_new/', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()


