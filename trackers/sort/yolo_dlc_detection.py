import numpy as np
import cv2
import math
import argparse
import time
import glob
import os
import torch
import traceback
import sys
from scipy.spatial import cKDTree
import pandas as pd

import yolo_detector
import dlc_detector

from sort import *

assembler_count = 0

def find_closest_neighbors(xy_true, xy_pred, k=5):
    n_preds = xy_pred.shape[0]
    tree = cKDTree(xy_pred)
    dist, inds = tree.query(xy_true, k=k)
    idx = np.argsort(dist[:, 0])
    neighbors = np.full(len(xy_true), -1, dtype=int)
    picked = set()
    for i, ind in enumerate(inds[idx]):
        for j in ind:
            if j not in picked:
                picked.add(j)
                neighbors[idx[i]] = j
                break
        if len(picked) == n_preds:
            break
    return neighbors

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def loadImage(path, img_size=640, stride=32, auto=True):
    # BGR
    img0 = cv2.imread(path)

    # Padded resize
    img = letterbox(img0, img_size, stride=stride, auto=auto)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    return img0.copy(), img.copy()


def check_double(bboxs, bpt_coord, bb_slack):
    is_double = False
    box_idx = np.where((bboxs[:,0]-bb_slack<= bpt_coord[0]) & (bboxs[:,1]-bb_slack<= bpt_coord[1]) & (bboxs[:,2]+bb_slack>= bpt_coord[0]) & (bboxs[:,3]+bb_slack>= bpt_coord[1]))[0]

    if np.shape(box_idx)[0] > 1:
        is_double = True

    return is_double,box_idx.copy()


'''
-- point: list of [[box IDs], body part, body part ID, x, y, conf]
-- Animal: numpy of [[x, y, confidence, body part, body part ID] * body parts]
-- points_costs: Dictionary with costs for each link in graph

'''
def compute_cost(point, animal, points_costs, graph):
    total_cost = 0

    for bpt in animal:
        if point[1] < bpt[3]:
            link = [int(point[1]), int(bpt[3])]
            low_ID = point[2]
            high_ID = bpt[4]
        else:
            link = [int(bpt[3]), int(point[1])]
            low_ID = bpt[4]
            high_ID = point[2]
        idx = graph.index(link)
        cost_arr = points_costs[idx]['m1']
        total_cost += cost_arr[int(low_ID), int(high_ID)]/animal.shape[0]
    return total_cost

# Match points that occur in more than one box with paf costs
def match_double_points(points_costs, animal_assemblies, double_points, graph, matched_points):

    for point in double_points:
        boxes = point[0]
        bpt = point[1]
        bpt_ID = point[2]
        x = point[3]
        y = point[4]
        conf = point[5]
        costs = np.array([])

        for box_id in boxes:
            animal = animal_assemblies[box_id]
            cost = -1
            # print(f'Animal {animal}')
            # print(f'Animal shape {animal.shape[0]}')
            if animal.shape[0] > 0: # TODO: Handle cases when bboxs have no connected keypoints yet (e.g if all points are in two boxes)
                if bpt not in animal[:,3]:
                    cost = compute_cost(point,animal,points_costs,graph)
            costs = np.append(costs,cost)



        if np.max(costs) != -1:
            max_id = boxes[np.argmax(costs)] # Add key point to animal with highest paf value
            animal_assemblies[max_id] = np.append(animal_assemblies[max_id], [[x , y, conf, bpt, bpt_ID]], axis = 0)
            matched_points[bpt].append(bpt_ID)
            # print(f'Animal assemblies: {animal_assemblies}')
            # print(f'Boxes: {boxes}')
            # print(f'Costs: {costs}')
            # print(f'Max id: {max_id}')
            # print(f'Animal: {animal_assemblies[max_id]}')
    return animal_assemblies, matched_points


# Maps keypoints that lies within a box to that box
# def match_keypoints2boxes(bboxs, points_coord, points_conf, points_costs, graph, frame, point_conf_thresh = 0.5, bb_slack = 0):
#     n_boxes = bboxs.shape[0]
#     n_bpts = len(points_coord)
#
#     #double_points = find_doubles(bboxs, points_coord)
#     #print(double_points)
#     double_points = []
#
#     animal_assemblies = []
#
#     matched_points = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
#
#     for box_idx, box in enumerate(bboxs):
#         animal = []
#         animal.append([])
#         for (bpt_idx, bpt_points), conf in zip(enumerate(points_coord), points_conf):
#             p_idx = np.where((bpt_points[:,0] >= box[0]-bb_slack) & (bpt_points[:,1] >= box[1]-bb_slack) & (bpt_points[:,0] <= box[2]+bb_slack) & (bpt_points[:,1] <= box[3]+bb_slack))[0] # Body parts in b.box
#
#             if not np.any(p_idx):
#                 continue
#
#             p_conf = conf[p_idx] # Confidence for body parts in b.box
#             if np.max(p_conf) >= point_conf_thresh:
#
#                 keep_idx = p_idx[np.argmax(p_conf)] # Keep point with highest confidence
#                 is_double, box_idx  = check_double(bboxs, bpt_points[keep_idx],bb_slack)
#
#                 if not is_double: # Check that the point does not occur in multiple bboxes
#                     animal.append([bpt_points[keep_idx,0],bpt_points[keep_idx,1],conf[keep_idx][0],bpt_idx,keep_idx]) # x, y, confidence, body part, body part ID
#                     matched_points[bpt_idx].append(keep_idx)
#                 else:  # Store points that occur in more than one bounding box, with box IDs, body part, and body part ID, x, y, conf
#                     double_points.append([box_idx.copy(), bpt_idx, keep_idx, bpt_points[keep_idx,0], bpt_points[keep_idx,1], conf[keep_idx][0]])
#         animal.pop(0)
#         animal_assemblies.append(np.array(animal))
#     if len(double_points) > 0:
#         #print(f'Double point at frame: {frame}')
#         animal_assemblies, matched_points = match_double_points(points_costs, animal_assemblies, double_points, graph, matched_points)
#         #print(double_points)
#
#     return animal_assemblies.copy(), matched_points.copy()

# Maps keypoints that lies within a box to that box
# TODO: Make sure that point is not matched twice. E.g check matched_points

def build_assemblies(assembler, bboxs, dlc_preds, graph, frame, point_conf_thresh = 0.5, bb_slack = 0):

    points_coord = dlc_preds['coordinates'][0]
    points_conf = dlc_preds['confidence']
    points_costs = dlc_preds['costs']

    animal_assemblies = []

    matched_points = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}

    assemblies, unique = assembler.assemble(dlc_preds, frame)
    if assemblies:
        animals = np.stack([ass.data for ass in assemblies])

        for animal in animals:

            animal_assemblies.append(animal.copy())

            for bpt_idx, point in enumerate(animal):

                if math.isnan(point[0]) or math.isnan(point[1]):
                    continue

                xy = [point[0], point[1]]
                bpt_preds = points_coord[bpt_idx].tolist()

                id = bpt_preds.index(xy)

                matched_points[bpt_idx].append(id)

    return animal_assemblies.copy(), matched_points.copy()

def match_keypoints2boxes(assembler, bboxs, dlc_preds, graph, frame, point_conf_thresh = 0.5, bb_slack = 0):

    points_coord = dlc_preds['coordinates'][0]
    points_conf = dlc_preds['confidence']
    points_costs = dlc_preds['costs']

    n_boxes = bboxs.shape[0]
    n_bpts = len(points_coord)

    #double_points = []
    animal_assemblies = []

    matched_points = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}

    for box_idx, box in enumerate(bboxs):
        #animal = []
        #animal.append([])

        # Dictionary with points in box as {0:[], 1:[], 2:[], 3:[], ... }
        bbox_points = {}
        for i in range(n_bpts):
            bbox_points[i] = []

        #print(f'Box: {box_idx}')

        for (bpt_idx, bpt_points), conf in zip(enumerate(points_coord), points_conf):
            p_idx = np.where((bpt_points[:,0] >= box[0]-bb_slack) & (bpt_points[:,1] >= box[1]-bb_slack) & (bpt_points[:,0] <= box[2]+bb_slack) & (bpt_points[:,1] <= box[3]+bb_slack))[0] # Body parts in b.box

            if p_idx.shape[0] == 0:
                continue

            p_idx = set(p_idx)

            # IDs for matched points
            #matched_id = set(matched_points[bpt_idx])

            # Choose points present in box but not already matched
            #keep_idx = list(p_idx.difference(matched_id))

            bbox_points[bpt_idx].extend(list(p_idx))

        global assembler_count
        assemblies, unique = assembler.assemble(dlc_preds.copy(), assembler_count, bbox_points = bbox_points.copy())
        assembler_count += 1

        if assemblies:
            animal = np.stack([ass.data for ass in assemblies])
            animal = animal.reshape(n_bpts,4)

            b_id = np.ones((n_bpts,1))*box_idx
            animal = np.append(animal, b_id, axis = 1)
            #print(animal)
            animal_assemblies.append(animal.copy())
            for bpt_idx, point in enumerate(animal):

                if math.isnan(point[0]) or math.isnan(point[1]):
                    continue


                if math.isnan(point[0]) or math.isnan(point[1]):
                    continue

                xy = [point[0], point[1]]

                bpt_preds = points_coord[bpt_idx].tolist()

                id = bpt_preds.index(xy)

                matched_points[bpt_idx].append(id)

                # bbox_bpt_id = bbox_points[bpt_idx] # ID for current bpt in bbox
                #
                # bpt_list = points_coord[bpt_idx]
                # bpt_list = bpt_list[bbox_bpt_id].tolist() # Points coord for bpts in bbox
                #
                # xy = [point[0], point[1]]
                # id = bbox_bpt_id[bpt_list.index(xy)]
                # matched_points[bpt_idx].append(id)

    return animal_assemblies.copy(), matched_points.copy()

def evaluate_assembled_data(gt_data, matched_points, pred_coord, pred_conf, joints, current_frame):
    tp_tot = 0
    fp_tot = 0
    fn_tot = 0
    bpt_result = []
    df = gt_data.unstack("coords").reindex(joints, level="bodyparts")

    for (bpt, xy_gt), bpt_match, bpt_preds, bpt_conf in zip(df.groupby(level="bodyparts"), matched_points.values(), pred_coord, pred_conf):

        # print(f'bpt: {bpt}')
        # print(f'gt: {xy_gt}')
        # All non-nan GT values
        inds_gt = np.flatnonzero(np.all(~np.isnan(xy_gt), axis=1))

        # Predicted xy and conf values
        xy = np.array(bpt_preds[bpt_match])
        conf = bpt_conf[bpt_match]

        if inds_gt.size and xy.size:

            # All non-nan GT values
            xy_gt_values = xy_gt.iloc[inds_gt].values
            #print('gt values:')
            #print(len(xy_gt_values))
            # Find closest neighbours for predictions
            neighbors = find_closest_neighbors(xy_gt_values, xy, k=3)
            found = neighbors != -1

            # Calculated distance between predictions and gt (rmse)
            min_dists = np.linalg.norm(xy_gt_values[found] - xy[neighbors[found]],axis=1)

            tp_tot += len(neighbors[found])
            fp_tot += len(xy) - len(neighbors[found])
            fn_tot += len(xy_gt_values) - len(neighbors[found])

            conf = conf[neighbors[found]].flatten()

            # Store results in dataframe
            bpt_df = pd.DataFrame({'rmse':min_dists, 'conf':conf})
            bpt_df['body part'] = bpt
            bpt_df['frame idx'] = current_frame
            bpt_result.append(bpt_df.copy())

        elif (inds_gt.size) and (not xy.size):
            xy_gt_values = xy_gt.iloc[inds_gt].values
            fn_tot += len(xy_gt_values)

        elif (not inds_gt.size) and (xy.size):
            fp_tot += len(xy)

    if len(bpt_result) > 0:
        eval_df = pd.concat(bpt_result)
        eval_df.reset_index(inplace = True, drop = True)
        return eval_df.copy(), tp_tot, fp_tot, fn_tot
    else:
        eval_df = None
        return eval_df, tp_tot, fp_tot, fn_tot

if __name__ == '__main__':
    display = False
    save_video = False
    evaluate = False
    save_images = False
    save_preds = True
    fps = 15
    width = 1920
    height = 1080

    count = 0
    bb_slack = 20

    colors = [(0,0,255),(0,255,0),(255,0,0),(128,0,128),(255,128,0),
        (255,105,180),(255,130,171),(187,255,255),(154,255,154),(255,255,0)]

    joints = ['head','beak','left_foot','left_wing','left_wing_tip','right_foot','right_wing','right_wing_tip','tail']

    video_path = 'C:\\Users\\MatMar\\Documents\\BirdMan_Project\\Videos\\BONDEN5'
    #video_name = 'Farallon3_20200517_105645_trim'
    video_name = 'BONDEN5_20210708_195930'
    video_format = '.avi'
    full_video_name = video_name+video_format
    full_video_path = os.path.join(video_path,full_video_name)

    output_folder = 'C:\\Users\\MatMar\\Documents\\BirdMan_Project\\Videos\\results'
    video_output_path = os.path.join(output_folder,video_name+'_detect'+video_format)
    output_image_folder = os.path.join(output_folder,"frames_new_assembler")

    yolo_weights = 'C:\\Users\\MatMar\\Documents\\BirdMan_Project\\model_weights\\yolov5l\\yolo_best.pt'

    # Image folder path

    # data_folder_path = "C:\\Users\\MatMar\\Documents\\BirdMan_Project\\resultat\\training_results\\B5_F3-2022-04-27"
    # data_shuffle = "B5_F3Apr20-trainset80shuffle1"
    # gt_data_path = os.path.join(data_folder_path, "training-datasets\\iteration-0\\UnaugmentedDataSet_B5_F3Apr20\\CollectedData_mathan.h5")

    data_folder_path = "C:\\Users\\MatMar\\Documents\\BirdMan_Project\\resultat\\training_results\\FARALLON3_22_04_29"
    data_shuffle = "FARALLON3Mar21-trainset80shuffle5"
    gt_data_path = os.path.join(data_folder_path, "training-datasets\\iteration-0\\UnaugmentedDataSet_FARALLON3Mar21\\CollectedData_mathan.h5")

    # data_folder_path = "C:\\Users\\MatMar\\Documents\\BirdMan_Project\\resultat\\training_results\\BONDEN5_22_04_27"
    # data_shuffle = "BONDEN5Mar14-trainset80shuffle5"
    # gt_data_path = os.path.join(data_folder_path, "training-datasets\\iteration-0\\UnaugmentedDataSet_BONDEN5Mar14\\CollectedData_mathan.h5")

    # DLC pose config
    pose_cfg_path = os.path.join(data_folder_path,"dlc-models\\iteration-0",data_shuffle,"test\\pose_cfg.yaml")

    # DLC inference config
    inference_cfg = os.path.join(data_folder_path,"dlc-models\\iteration-0",data_shuffle,"test\\inference_cfg.yaml")

    Data = pd.read_hdf(gt_data_path)

    # Reference to paths to all frames in data folder
    frame_paths_list = list(Data.index.to_list())

    if evaluate:
        frame_eval_results = []
        metrics_data = []

    if not os.path.exists(full_video_path):
        print('\n\tERROR: Path to video not found!')
        exit()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if save_video:
        output_video = cv2.VideoWriter(video_output_path,cv2.VideoWriter_fourcc('F','M','P','4'), fps,(width,height))

    # Save predictions
    tot_save_preds = []

    # Load yolov5 detector model
    yolo_model = yolo_detector.Detector(yolo_weights)

    # Load dlc detector models
    dlc_model = dlc_detector.Detector(pose_cfg_path,inference_cfg)
    graph = dlc_model.get_graph()
    assembler = dlc_model.get_assembler()
    assembler.set_max_individuals(10)

    # Load dataset using Yolov5 dataloader.
    dataset = yolo_model.load_data2model(full_video_path)

    print("Processing video")

    #for current_frame, (path, im, im0, vid_cap, s) in enumerate(dataset):
    for current_frame, path_list in enumerate(frame_paths_list):
        print(f'Processing frame: {current_frame}', end = '\r')
        frame_path = os.path.join(data_folder_path, os.path.join(*path_list))
        im0, im = loadImage(frame_path)

        # if current_frame > 10:
        #     break

        # Make predictions on image
        yolo_dets =  yolo_model.predict_next(im,im0)

        pred_coord, pred_conf, pred_costs, dlc_dets, animals, xy =  dlc_model.predict_next(im0)
        try:

            # animal_assemblies, matched_points = match_keypoints2boxes(assembler, yolo_dets, dlc_dets, graph, current_frame, point_conf_thresh = 0.8, bb_slack=bb_slack)

            animal_assemblies, matched_points = build_assemblies(assembler, yolo_dets, dlc_dets, graph, current_frame, point_conf_thresh = 0.8, bb_slack = bb_slack)

        except Exception:
            traceback.print_exc()
            print('Closing and ending session')
            break

        if evaluate:
            GT = Data.iloc[current_frame]
            if GT.any():
                frame_eval_df, tp, fp, fn = evaluate_assembled_data(GT, matched_points, pred_coord, pred_conf, joints, current_frame)
                if frame_eval_df is not None:
                    frame_eval_results.append(frame_eval_df.copy())
                    metrics_data.append([ tp, fp, fn, current_frame])
                #print(f'True positives: {tp}, False positives: {fp}, False negatives: {fn}')


        if(display or save_video or save_images):

            for d in yolo_dets:
                d = d.astype(np.int32)
                #rec = cv2.rectangle(im0,(d[0], d[1]),(d[2],d[3]),(0, 0, 255), 2)
                rec = cv2.rectangle(im0,(d[0]-bb_slack, d[1]-bb_slack),(d[2]+bb_slack,d[3]+bb_slack),(0, 255, 0), 2)

            if animal_assemblies:
                for i, bird in enumerate(animal_assemblies):
                    c_idx = i%len(colors)-1
                    for bpt in bird:
                        if math.isnan(bpt[0]):
                            continue
                        circ = cv2.circle(im0,(int(bpt[0]),int(bpt[1])), 5, colors[c_idx], -1)
                        # im0 = cv2.putText(circ,f'{conf[0]:.3f}', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[c_idx], 4)

        if save_preds and animal_assemblies:
            preds = []
            for i, bird in enumerate(animal_assemblies):
                for bpt_idx, bpt in enumerate(bird):
                    if math.isnan(bpt[0]):
                        continue
                    preds.append([bpt[0],bpt[1],joints[bpt_idx],f'bird{i}',current_frame])
            tot_save_preds.extend(preds.copy())


        if save_images:
            output_file_path = os.path.join(output_image_folder,f'img{current_frame}.jpg')
            saved = cv2.imwrite(output_file_path,im0)

        if(display):
            cv2.imshow('image',im0)
            cv2.waitKey(1)

        if save_video:
            output_video.write(im0)

        current_frame += 1

    preds_df = pd.DataFrame(tot_save_preds, columns = ['x','y','bodyparts','individual','frame'])
    print(preds_df)

    if save_preds:
        preds_df.to_csv(os.path.join(data_folder_path,"assembler_preds_shuffle.csv"), index=False)

    if evaluate:
        tot_eval_data = pd.concat(frame_eval_results)
        tot_eval_data.reset_index(inplace = True, drop = True)
        print(tot_eval_data)
        tot_eval_data.to_csv(os.path.join(data_folder_path,"assembler_eval_shuffle.csv"), index=False)
        metrics_df = pd.DataFrame(metrics_data, columns = ['tp', 'fp', 'fn', 'frame'])
        print(metrics_df)
        metrics_df.to_csv(os.path.join(data_folder_path,"assembler_metrics_shuffle.csv"), index=False)
    if save_video:
        output_video.release()

    if display:
        cv2.destroyAllWindows()

    dlc_model.close()
    print(f"Total detection of {current_frame} frames")
