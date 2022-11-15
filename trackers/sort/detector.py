import torch
import os
import torchvision.transforms as transforms
import numpy as np
from yolov5_utils import non_max_suppression
from yolov5.detect import run



class Detector():
    """
    This class represents the detector model
    """
    def __init__(self,model_type,weights = None):
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(size = (640,640))])
        if model_type == 'yolov5l':
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l',pretrained = True,classes = 3)
        else:
            print(f'Model type: {model_type} not supported')
            exit()
        if weights is not None:
            if os.path.exists(weights):
                checkpoint = torch.load(weights)['model']
                self.model.model.load_state_dict(checkpoint.state_dict())
            else:
                print('Could not find model weights')
                exit()   
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        #self.model.eval()
        self.model.to(self.device)
             

    def detect(self,img):
        """Predict objects in given frame"""
        img = self.transform(img)
        img = img.unsqueeze(0)
        img.to(self.device)
        detections = self.model(img)
        print('------------Detections-------------')
        detections = non_max_suppression(detections[0])[0].detach()
        print(detections)
        # Convert detections to [x1,y1,x2,y2]
        detections = np.delete(detections.numpy(),-1,axis=1)
        
        return detections




# class Detector():
#     """
#     This class represents the detector model
#     """
#     def __init__(self,model_type,weights = None):
#         self.transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(size = (640,640))])
#         if model_type == 'yolov5l':
#             self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l',pretrained = True,classes = 3)
#         else:
#             print(f'Model type: {model_type} not supported')
#             exit()
#         if weights is not None:
#             if os.path.exists(weights):
#                 checkpoint = torch.load(weights)['model']
#                 self.model.model.load_state_dict(checkpoint.state_dict())
#             else:
#                 print('Could not find model weights')
#                 exit()   
#         if torch.cuda.is_available():
#             self.device = torch.device("cuda")
#         else:
#             self.device = torch.device("cpu")
#         #self.model.eval()
#         self.model.to(self.device)
             

#     def detect(self,img):
#         """Predict objects in given frame"""
#         img = self.transform(img)
#         img = img.unsqueeze(0)
#         img.to(self.device)
#         detections = self.model(img)
#         print('------------Detections-------------')
#         detections = non_max_suppression(detections[0])[0].detach()
#         print(detections)
#         # Convert detections to [x1,y1,x2,y2]
#         detections = np.delete(detections.numpy(),-1,axis=1)
        
#         return detections
