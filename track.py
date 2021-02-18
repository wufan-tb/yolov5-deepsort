import argparse
from skimage import draw
import torch.backends.cudnn as cudnn
from models.experimental import *
from utils.datasets import *
from utils.utils import *
import sys
from tqdm import tqdm
from utils.post_process import *
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


def bbox_rel(image_width, image_height,  *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)
    
def draw_boxes(img, bbox, labels, identities=None, v=None):
    for i, box in enumerate(bbox):
        xmin, ymin, xmax, ymax = [int(i) for i in box]
        ymin = min(img.shape[0]-5,max(5,ymin))
        xmin = min(img.shape[1]-5,max(5,xmin))
        ymax = max(5,min(img.shape[0]-5,ymax))
        xmax = max(5,min(img.shape[1]-5,xmax))
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label=labels[i]
        v=v if v is not None else 0
        v_real=velocity_transform(v[i],ymax,350,H=6,L1=2,L2=50)
        info = '{:d}|{:.1f}km/s'.format(id,v_real)
        t_size=cv2.getTextSize(info, cv2.FONT_HERSHEY_TRIPLEX, 0.4 , 1)[0]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.rectangle(img, (xmin, ymin), (xmin + t_size[0]+2, ymin + t_size[1]+4), color, -1)
        cv2.putText(img, info, (xmin+1, ymin+t_size[1]+1), cv2.FONT_HERSHEY_TRIPLEX, 0.4, [255,255,255], 1)
    return img

def velocity_transform(v_pix,pixel_y,pixel_ymax,H,L1,L2):
    theta1=math.atan(L1/H)
    theta2=math.atan(L2/H)
    theta=theta1+(theta2-theta1)*((pixel_ymax-pixel_y)/pixel_ymax)
    rate=H/((math.cos(theta))**2)
    return rate*v_pix

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('rtmp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    #spetical set
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT, 
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, 
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, 
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, 
                        use_cuda=True, use_appearence=True)
    if source.endswith('.mp4') or source.endswith('.avi'):
        fps=25
        output_path=source[0:-4]+'_demo.avi'
        fourcc = cv2.VideoWriter_fourcc('P','I','M','1')
        cap = cv2.VideoCapture(source)
        _, frame = cap.read()
        cap.release()
        my_video = cv2.VideoWriter(output_path, fourcc, fps ,(frame.shape[1],frame.shape[0]))
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
            
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in tqdm(enumerate(pred)):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            current_frame=[[],[],[]]
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    
                bbox_xywh = []
                confs = []
                labels = []
                # Write results using deep sort
                for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    labels.append(int(cls))
                
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                start=time.time()
                outputs = deepsort.update(xywhs, confss , labels, im0)
                print('track time:',time.time()-start)
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    labels = outputs[:, 4]
                    label_names=[names[int(item)] for item in labels]
                    identities = outputs[:, 5]
                    v=np.sqrt((outputs[:, 6]/10)**2+(outputs[:, 7]/10)**2)
                    im0=draw_boxes(im0, bbox_xyxy, label_names, identities,v)
            
            my_video.write(im0)
            print('-----------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5l.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='D:/dataset/video/1.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='./inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 1 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
