from __future__ import print_function

from sort import *
import argparse
import time
import glob
import os
import numpy as np
import cv2
import torch

import pandas as pd

import yolo_detector
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display',
                        help='Display online tracker output (slow) [False]', action='store_true')
    parser.add_argument("--video_path", help="Path to video to analyse.",
                        type=str, default='C:\\Users\\User\\Documents\\CAS\\Ex_jobb\Data\\video\\')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=4)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=5)
    parser.add_argument("--iou_threshold",
                        help="Minimum IOU for match.", type=float, default=0.3)
    parser.add_argument("--out_path", help="Path of output dir.",
                        type=str, default='runs')
    parser.add_argument("--weights", help="Path to yolo weights.",
                        type=str, default='../../trained_detector/yolov5/weights/exp24weights.pt')
    parser.add_argument(
        "--csv",
        default="",
        type=str,
        help="CSV with bbox gt",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    display = args.display
    save_images = False
    save_video = True
    fps = 25
    width = 1920
    height = 1080

    # video_path = '/home/Data/videos'
    full_video_path = args.video_path
    video_name = full_video_path.split('/')[-1].split('.')[0]
    #video_name = 'Farallon3_20190513_151447_No005'
    #video_name = 'BONDEN5_20210710_145319_fighting'
    # video_name = 'Farallon3_20200517_105645_trim'
    video_format = '.avi'
    # full_video_name = video_name+video_format
    # full_video_path = os.path.join(video_path,full_video_name)

    #output_folder = 'C:\\Users\\User\\Documents\\CAS\\Ex_jobb\\Data\\results\\yolo_sort_output'
    output_folder = args.out_path
    os.makedirs(output_folder, exist_ok=True)
    output_image_folder = os.path.join(output_folder,video_name)
    video_output_path = os.path.join(output_folder,video_name+'_tracked'+video_format)
    det_video_output_path = os.path.join(output_folder,video_name+'_detect'+video_format)
    track_txt = os.path.join(output_folder, video_name+'.txt')
    txt = ''

    if len(args.csv):
        print(args.csv)
        det_df = pd.read_csv(args.csv, names=['frame_id','x1', 'y1', 'w', 'h'])


    weights = args.weights

    total_time = 0.0

    if not os.path.exists(full_video_path):
        print('\n\tERROR: Path to video not found!')
        exit()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)


    # Tracker model. Creates instance of the SORT tracker
    mot_tracker = Sort(max_age=args.max_age,
                        min_hits=args.min_hits,
                        iou_threshold=args.iou_threshold)

    # Detector model
    detector = yolo_detector.Detector(weights)

    # Dataloader. Load data to detector model
    dataset = detector.load_data2model(full_video_path)

    if save_video:
        tracked_video = cv2.VideoWriter(video_output_path,cv2.VideoWriter_fourcc('F','M','P','4'), fps,(width,height))
        #detected_video = cv2.VideoWriter(det_video_output_path,cv2.VideoWriter_fourcc('F','M','P','4'), fps,(width,height))

    print("Processing video")

    current_frame = 1
    for path, im, im0s, vid_cap, s in dataset:

        # Make predictions on image
        if len(args.csv):
            frame_df = det_df[det_df['frame_id']==current_frame][['x1', 'y1', 'w', 'h']].copy()
            frame_df.w += frame_df.x1
            frame_df.h += frame_df.y1
            frame_df['conf'] = [0.99] * len(frame_df)
            dets = torch.from_numpy(frame_df.to_numpy())
        else:
            dets =  detector.predict_next(im,im0s)
        

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        if(display or save_images or save_video):
            for d in trackers:
                if not len(args.csv):
                    d = d.astype(np.int32)
                rec = cv2.rectangle(im0s,(int(d[0]), int(d[1])),(int(d[2]),int(d[3])),(0, 255, 0), 4)
                im0s = cv2.putText(rec,f'Bird {d[4]}', (int(d[0]),int(d[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 4)
                txt += f'{current_frame},{d[4]},{d[0]},{d[1]},{d[2]-d[0]},{d[3]-d[1]},0.7,-1,-1.-1\n'
            for d in dets:
                if not len(args.csv):
                    d = d.astype(np.int32)
                rec = cv2.rectangle(im0s,(int(d[0]), int(d[1])),(int(d[2]),int(d[3])),(0, 0, 255), 2)

        if(display):
            cv2.imshow('image',im0s)
            cv2.waitKey(1)

        if save_images:
            output_file_path = os.path.join(output_image_folder,f'img{current_frame}.jpg')
            saved = cv2.imwrite(output_file_path,im0s)

        if save_video:
            tracked_video.write(im0s)

        

        print(f'Processing frame: {current_frame}',end = '\r')
        current_frame += 1

    if save_video:
        tracked_video.release()

    if display:
        cv2.destroyAllWindows()

    with open(track_txt,'w+') as f:
        f.write(txt)

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" %
          (total_time, current_frame, current_frame / total_time))
