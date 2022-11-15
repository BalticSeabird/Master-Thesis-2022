import torch
import sys
import numpy as np

#sys.path.append('/home/Seabird_Master_Thesis/trained_detector/yolov5')
sys.path.append('/home/tracking_branch/Seabird_Master_Thesis/trained_detector/yolov5')
from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, print_args, scale_coords
from utils.torch_utils import select_device

class Detector():
    """
    This class represents the yolo detector model for tracking
    with SORT
    """
    def __init__(self,weights,imgsz=(640, 640),conf_thres=0.40,iou_thres=0.20):

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            print(f'Device {device}')
            self.device = select_device(device)
        else:
            self.device = select_device('cpu')

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.model = DetectMultiBackend(weights, device=self.device)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)

        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))

    def load_data2model(self,data_path,batch_size = 1):
        self.bs = batch_size
        self.dataset = LoadImages(data_path, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        return self.dataset

    def predict_next(self,im,im0):
        im = self.prep_data(im)
        # Inference
        pred = self.model(im, augment=False, visualize=False)
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)[0]
        out = []
        if len(pred):
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], im0.shape).round()
            # Format results
            for *xyxy, conf, cls in reversed(pred):
                tmp_out = torch.tensor(xyxy).tolist()
                tmp_out.append(conf.item())
                out.append(tmp_out)
        dets = np.array(out.copy())
        return dets


    def prep_data(self,im):
        im = torch.from_numpy(im).to(self.device)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im
