import sys
import numpy as np
import tensorflow as tf
import os
from skimage.util import img_as_ubyte
from collections import defaultdict # Check what this does
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

from live_dlc_assembler import Assembler

# sys.path.append('/home/Seabird_Master_Thesis/trained_detector/yolov5')
# from utils.datasets import LoadImages

#sys.path.append('/home/Seabird_Master_Thesis/trained_detector/DeepLabCut')
sys.path.append('C:\\Users\\MatMar\\Documents\\BirdMan_Project\\Seabird_Master_Thesis\\trained_detector\\DeepLabCut')
from deeplabcut.pose_estimation_tensorflow.core import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict_multianimal as predictma
from deeplabcut.pose_estimation_tensorflow.util import visualize
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils
from deeplabcut.pose_estimation_tensorflow.lib import trackingutils

class Detector():
    """
    This class represents the dlc detector model for tracking
    with SORT
    """
    def __init__(self,path_test_config,path_inference_config,gputouse = 1,n_bpts = 9,n_ind = 10,pcutoff = 0.1):
        dlc_cfg = load_config(str(path_test_config))
        inference_cfg = load_config(path_inference_config)

        self.n_multibodyparts = n_bpts
        self.pcutoff = pcutoff
        self.max_n_individuals = n_ind

        self.boundingboxslack = inference_cfg['boundingboxslack']

        graph = dlc_cfg["partaffinityfield_graph"]
        #limbs = dlc_cfg.get("paf_best", np.arange(len(graph)))
        limbs = np.arange(len(graph))
        graph = [graph[l] for l in limbs]

        #self.ass = Assembler(max_n_individuals = self.max_n_individuals, n_multibodyparts = self.n_multibodyparts, greedy = True, graph=graph, paf_inds=limbs)
        self.ass = Assembler(max_n_individuals = 10, n_multibodyparts = self.n_multibodyparts, greedy = True, min_affinity=0.01, pcutoff = 0.6, graph=graph, paf_inds=limbs)

        self.paf_inds = limbs
        self.graph = graph
        self.dlc_cfg = dlc_cfg

        self.frame_count = 0

        self.sess, self.inputs, self.outputs = predict.setup_pose_prediction(self.dlc_cfg)
        if gputouse is not None:  # gpu selection
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gputouse)
        tf.compat.v1.reset_default_graph()

    def get_graph(self):
        return self.graph.copy()

    def get_assembler(self):
        return self.ass
    # Predicts and returns detections from model as a np array
    def predict_next(self,im):

        frame = img_as_ubyte(im.copy())
        frame = np.expand_dims(frame, axis=0)

        frame_ind = self.frame_count
        frame_ind += 1

        preds = predictma.predict_batched_peaks_and_costs(self.dlc_cfg,frame,self.sess, self.inputs, self.outputs)

        pred_coord = None
        pred_conf = None
        animals = None
        xy = None

        if preds:
            dets = preds[0]

            pred_coord = dets['coordinates'][0]
            pred_conf = dets['confidence']
            pred_costs = dets['costs']

            # assemblies, unique = self.ass.assemble(dets, frame_ind)
            #
            # if assemblies:
            #     animals = np.stack([ass.data for ass in assemblies])
            #     xy = trackingutils.calc_bboxes_from_keypoints(animals,self.boundingboxslack)

        self.frame_count = frame_ind

        return pred_coord, pred_conf, pred_costs, dets, animals, xy

    def close(self):
        self.sess.close()
