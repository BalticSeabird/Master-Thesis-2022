import numpy as np
import cv2
import math
#import torch
import traceback
import sys

import dlc_detector

sys.path.append('C:\\Users\\MatMar\\Documents\\BirdMan_Project\\Seabird_Master_Thesis\\trained_detector\\DeepLabCut')

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.lib import trackingutils

if __name__ == '__main__':

    #pose_cfg_path = '/home/FARALLON3-mathan-2022-03-21/dlc-models/iteration-0/FARALLON3Mar21-trainset95shuffle1/test/pose_cfg.yaml'
    pose_cfg_path = 'C:\\Users\\MatMar\\Documents\\BirdMan_Project\\resultat\\B5_F3-2022-04-20\\dlc-models\\iteration-0\\B5_F3Apr20-trainset95shuffle1\\test\\pose_cfg.yaml'
    #full_video_path = '/home/Data/videos/Farallon3_20200517_105645_trim.avi'
    full_video_path = 'C:\\Users\\MatMar\\Documents\\BirdMan_Project\\Videos\\Farallon3\\Farallon3_20200517_105645_trim.avi'
    # Inference config
    inference_cfg = 'C:\\Users\\MatMar\\Documents\\BirdMan_Project\\resultat\\B5_F3-2022-04-20\\dlc-models\\iteration-0\\B5_F3Apr20-trainset95shuffle1\\test\\inference_cfg.yaml'
    # Detector model
    detector = dlc_detector.Detector(pose_cfg_path,inference_cfg)

    # Dataloader. Load data to detector model
    #dataset = detector.load_data2model(full_video_path)
    cap = cv2.VideoCapture(full_video_path)

    fps = 25
    width = 1920
    height = 1080

    video_output_path = 'C:\\Users\\MatMar\\Documents\\BirdMan_Project\\Videos\\Farallon3\\Farallon3_20200517_105645_trim_keypoints.avi'

    tracked_video = cv2.VideoWriter(video_output_path,cv2.VideoWriter_fourcc(*'XVID'), fps,(width,height))

    count = 0

    colors = [(0,0,255),(0,255,0),(255,0,0),(128,0,128),(255,128,0),
        (255,105,180),(255,130,171),(187,255,255),(154,255,154),(255,255,0)]

    while(cap.isOpened()):
        ret, frame = cap.read()
        if count >= 500:
            break

        if ret == False:
            break

        try:
            # Make predictions on image
            pred_coord, pred_conf, animals, xy =  detector.predict_next(frame)
        except Exception:
            traceback.print_exc()
            print('Closing and ending session')
            break

        # if animals is not None:
        #     for coord in xy:
        #         x1 = int(coord[0])
        #         y1 = int(coord[1])
        #         x2 = int(coord[2])
        #         y2 = int(coord[3])
        #         rec = cv2.rectangle(frame,(x1,y1),(x2,y2),(0, 255, 0), 4)
        #
        #     for i, bird in enumerate(animals):
        #         c_idx = i%len(colors)-1
        #         for point in bird:
        #             if not math.isnan(point[3]):
        #                 x = int(point[0])
        #                 y = int(point[1])
        #                 cv2.circle(frame,(x,y), 5, colors[c_idx], -1)

        if pred_coord is not None:
            for i, bpt_coords in enumerate(pred_coord):
                c_idx = i%len(colors)-1
                for j, coord in enumerate(bpt_coords):
                    conf = pred_conf[i][j]
                    if conf >= 0.5:
                        x = int(coord[0])
                        y = int(coord[1])
                        circ = cv2.circle(frame,(x,y), 5, colors[c_idx], -1)
                        frame = cv2.putText(circ,f'{conf}', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[c_idx], 4)

        cv2.imshow('im',frame)
        #tracked_video.write(frame)
        cv2.waitKey(1)

        count += 1

    tracked_video.release()
    cv2.destroyAllWindows()
    print(f'total frames: {count}')
    detector.close()
