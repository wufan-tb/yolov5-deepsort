import argparse
from skimage import draw
import torch.backends.cudnn as cudnn
from models.experimental import *
from utils.datasets import *
from utils.utils import *
import sys
from tqdm import tqdm
sys.path.append('../')
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

def between(x,x_min,x_max):
    return min(x_max,max(x,x_min))

def fill_box(box_list,mask,fill_size=25):
    for box in box_list:
        cenXY=[(box[0]+box[2])/2,(box[1]+box[3])/2]
        cenXY=[between(cenXY[0],0+fill_size,mask.shape[1]-fill_size),between(cenXY[1],0+fill_size,mask.shape[0]-fill_size)]
        Y=np.array([cenXY[1]-fill_size,cenXY[1]-fill_size,cenXY[1]+fill_size,cenXY[1]+fill_size])
        X=np.array([cenXY[0]-fill_size,cenXY[0]+fill_size,cenXY[0]+fill_size,cenXY[0]-fill_size])
        yy, xx=draw.polygon(Y,X)
        mask[yy, xx] = 255
    return mask

def value_to_color(grayimg,low_value=15,high_value=220,low_color=[10,10,10],high_color=[255,10,10]):
    cv2.imwrite('gray.jpg',grayimg)
    r=low_color[0]+((grayimg-low_value)/(high_value-low_value))*(high_color[0]-low_color[0])
    g=low_color[1]+((grayimg-low_value)/(high_value-low_value))*(high_color[1]-low_color[1])
    b=low_color[2]+((grayimg-low_value)/(high_value-low_value))*(high_color[2]-low_color[2])
    # cv2.imwrite('r.jpg',r)
    # cv2.imwrite('g.jpg',g)
    # cv2.imwrite('b.jpg',b)
    rgb=np.ones((grayimg.shape[0],grayimg.shape[1],3))
    rgb[:,:,0]=r
    rgb[:,:,1]=g
    rgb[:,:,2]=b
    # cv2.imwrite('rgb.jpg',rgb)
    return rgb.astype('uint8')
              
def draw_obj_dense(img,box_list,k_size=281,beta=1.5):
    value=np.ones((img.shape[0],img.shape[1])).astype('uint8')
    value=value*10
    value=fill_box(box_list,value)
    value=cv2.GaussianBlur(value, ksize=(k_size,k_size),sigmaX=0,sigmaY=0)
    color=value_to_color(value)
    color=cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
    value[value<=20]=0.9
    value[value>20]=1.0
    mask=np.ones_like(img)
    mask[:,:,0]=value
    mask[:,:,1]=value
    mask[:,:,2]=value
    mask_color=mask*color
    mask_color=cv2.GaussianBlur(mask_color, ksize=(7,7),sigmaX=0,sigmaY=0)
    result = cv2.addWeighted(img, 1, mask_color, beta, 0)
    info='Total number: {}'.format(len(box_list))
    W_size,H_size=cv2.getTextSize(info, cv2.FONT_HERSHEY_TRIPLEX, 0.8 , 2)[0]
    cv2.putText(result, info, (3, 1+H_size+9), cv2.FONT_HERSHEY_TRIPLEX, 0.8, [0,255,0], 2)
    return result
    
def draw_boxes(img, bbox, labels, identities=None, Vx=None, Vy=None):
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
        vx=Vx[i] if Vx is not None else 0
        vy=Vy[i] if Vy is not None else 0
        info = '{:d}'.format(id)
        t_size=cv2.getTextSize(info, cv2.FONT_HERSHEY_TRIPLEX, 0.4 , 1)[0]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.rectangle(img, (xmin, ymin), (xmin + t_size[0]+2, ymin + t_size[1]+4), color, -1)
        cv2.putText(img, info, (xmin+1, ymin+t_size[1]+1), cv2.FONT_HERSHEY_TRIPLEX, 0.4, [255,255,255], 1)
    return img
    
def draw_counter(img,counter_dict):
    color=[255,125,125]
    for i in range(len(counter_dict.keys())+1):
        if i==0:
            info='Direction  '
            half_W=cv2.getTextSize(info, cv2.FONT_HERSHEY_TRIPLEX, 0.8 , 2)[0][0]
            for key in counter_dict[list(counter_dict.keys())[0]]:
                info+=' '+key+' '
        else:
            title=list(counter_dict.keys())[i-1]
            number=''
            for key in counter_dict[list(counter_dict.keys())[i-1]]:
                number+='  '+str(counter_dict[list(counter_dict.keys())[i-1]][key])+' '
        W_size,H_size=cv2.getTextSize(info, cv2.FONT_HERSHEY_TRIPLEX, 0.8 , 2)[0]
        
        if i==0:
            cv2.rectangle(img, (1, 1), (1 + W_size+4, 1 + (len(counter_dict.keys())+1)*(H_size+12)), color, -1)
            cv2.putText(img, info, (1+2, 1+(i+1)*(H_size+9)), cv2.FONT_HERSHEY_TRIPLEX, 0.8, [255,255,255], 2)
        else:
            cv2.putText(img, title, (1+2, 1+(i+1)*(H_size+9)), cv2.FONT_HERSHEY_TRIPLEX, 0.8, [255,255,255], 2)
            cv2.putText(img, number, (1+2+half_W, 1+(i+1)*(H_size+9)), cv2.FONT_HERSHEY_TRIPLEX, 0.8, [255,255,255], 2)
    return img

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
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)
    
    draw_info={}
    #draw_info['label_name']= {0:'person', 1:'motorbike', 2:'car', 3:'bicycle'}
    draw_info['label_name']= {0:'person', 2:'car', 5:'bus', 7:'truck'}
    draw_info['label_color']={0:[32,178,170], 1:[0,255,0],2:[255,0,0],3:[0,255,255],4:[255,255,0],5:[255,0,255],7:[0,0,255]}
    draw_info['draw_threshold']={0:0.5,1:0.5,2:0.3,3:0.5,4:0.5,5:0.5,6:0.5,7:0.5}
    draw_info['box_type']=['xmin','ymin','xmax','ymax']
    
    quyu_path=opt.quyu_img
    quyu_img = cv2.imread(quyu_path)
    quyu_img = cv2.cvtColor(quyu_img, cv2.COLOR_BGR2GRAY)
    _, quyu_img = cv2.threshold(quyu_img, 175, 255, cv2.THRESH_BINARY_INV)
    Area_list = np.argwhere(quyu_img > 250).tolist()
        
    Vector={'In':np.array([2,10]),'Out':np.array([-4,-10]),'All':np.array([0,0])}
    Last_ID={key:[] for key in Vector.keys()}
    Counter={draw_info['label_name'][key]:{label_key:0 for label_key in Vector.keys()} for key in opt.classes}
    if source.endswith('.mp4') or source.endswith('.avi'):
        fps=25
        output_path=source[0:-4]+'_demo.avi'
        fourcc = cv2.VideoWriter_fourcc('P','I','M','1')
        my_video = cv2.VideoWriter(output_path, fourcc, fps ,(1270,720))
    
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
                #print('track time:',time.time()-start)
                # draw boxes for visualization
                size_k=quyu_img.shape[0]/im0.shape[0]
                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    labels = outputs[:, 4]
                    identities = outputs[:, 5]
                    Vx = outputs[:, 6]/10
                    Vy = outputs[:, 7]/10
                    for i in range(len(outputs)):
                        box=bbox_xyxy[i]
                        id=identities[i]
                        label=labels[i]
                        center_coords=[int(size_k*box[3]),int(size_k*(box[0]+box[2])/2)]
                        vx=Vx[i]
                        vy=Vy[i]
                        if (center_coords in Area_list):
                            if id not in Last_ID['All']:
                                Counter[draw_info['label_name'][label]]['All'] += 1
                                Last_ID['All'].append(id)
                                if len(Last_ID['All'])>20:
                                    Last_ID['All']=Last_ID['All'][10:]
                            if (np.sqrt(vx**2+vy**2)>4):
                                for key in Vector.keys():
                                    if key == 'All':
                                        continue
                                    V=np.array([vx,vy])
                                    theta=np.dot(V,Vector[key])/(np.sqrt(np.dot(V,V))*np.sqrt(np.dot(Vector[key],Vector[key])))
                                    if  theta>0.6 and id not in Last_ID[key]:
                                        Counter[draw_info['label_name'][label]][key] += 1
                                        Last_ID[key].append(id)
                                        if len(Last_ID[key])>20:
                                            Last_ID[key]=Last_ID[key][10:]
                                        break
                    label_names=[draw_info['label_name'][labels[i]] for i in range(len(labels))]
                    im0=draw_boxes(im0, bbox_xyxy, label_names, identities, Vx, Vy)
                    im0=draw_obj_dense(im0,bbox_xyxy)
            #im0=draw_counter(im0,Counter)
            #im0=draw_bounding(im0,quyu_path)
            my_video.write(im0)
            print('-----------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5l.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./inference/images/shortone.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='./inference/output', help='output folder')  # output folder
    parser.add_argument('--quyu_img', default='./inference/quyu/all.jpg', help='quyu img path')
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
