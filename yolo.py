import torch.backends.cudnn as cudnn
from utils import google_utils
from utils.datasets import *
from utils.utils import *
import time

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



class YOLO(object):
    
    def __init__(self):
    
        self.model_path = 'model_data/yolov5s.pt'
        print("test")
        self.device = torch_utils.select_device('cpu')
        print(self.device)
        print("aa")
        self.half = self.device.type != 'cpu'
        print("aaaaa")
        self.model = torch.load(self.model_path, map_location = self.device)['model'].float().eval()
        print("Done")
        if self.half:
            self.model.half()
        self.img_size = 608
        self.iou_thres = 0.5
        self.conf_thres = 0.5 
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
    def detect_image(self, img):
        print(img.shape)
        gn = torch.tensor(img.shape)[[1, 0, 1, 0]]
        img_cp = letterbox(img, new_shape = self.img_size)[0]
        print(img_cp.shape)
        img_cp = img_cp[:, :, : : -1].transpose(2, 0, 1)
        img_cp = np.ascontiguousarray(img_cp)
        img_cp = torch.from_numpy(img_cp).to(self.device)
        img_cp = img_cp.half() if self.half else img_cp.float()
        img_cp /= 255.0
        if img_cp.ndimension() == 3:
            img_cp = img_cp.unsqueeze(0)
        pred = self.model(img_cp)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes = None, agnostic = None)
        boxes = []
        scores = []
        classes = []
        for i, det in enumerate(pred):
            
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_cp.shape[2:], det[:, :4], img.shape).round()
                
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    print("%g %ss, " %(n, self.names[int(c)]))
                
                for *xyxy, conf, clas in det:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    print(xywh)
                    if clas == 0 or clas == 1:
                        boxes.append([xywh[0], xywh[1], xywh[2], xywh[3]])
                        scores.append(conf)
                        classes.append(clas)
        return boxes, scores, classes
                

                    
            
        
        
            
        
        

    

        
        
                
            
            
        
