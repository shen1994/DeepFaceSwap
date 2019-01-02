# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:30:45 2018

@author: shen1994
"""

import cv2
import torch
import face_detect
import face_alignment
import numpy as np
import tensorflow as tf
from umeyama import umeyama

class FaceProcess:
    
    def __init__(self, use_fa):
        self.pnet = None
        self.rnet = None
        self.onet = None
        self.use_fa = use_fa
        self.fa = None
        self.image_label = 0
        self.ratio_landmarks = [
            (0.31339227236234224, 0.3259269274198092),
            (0.31075140146108776, 0.7228453709528997),
            (0.5523683107816256, 0.5187296867370605),
            (0.7752419985257663, 0.37262483743520886),
            (0.7759613623985877, 0.6772957581740159)] 
        
    def load_models(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = face_detect.create_mtcnn(sess, None)
            
        if self.use_fa:
            if torch.cuda.is_available():
                self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
                                                  device='cuda', flip_input=False)
            else:
                self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
                                                  device='cpu', flip_input=False)

    def process_bbox(self, bboxes, image_shape):
        
        for i, bbox in enumerate(bboxes):
            y0, x0, y1, x1 = bboxes[i, 0:4]
            w, h = int(y1 - y0), int(x1 - x0)
            length = (w + h) / 2
            center = (int((x1+x0)/2), int((y1+y0)/2))
            new_x0 = np.max([0, center[0]-length//2])
            new_x1 = np.min([image_shape[0], center[0]+length//2])
            new_y0 = np.max([0, center[1]-length//2])
            new_y1 = np.min([image_shape[1], center[1]+length//2])
            bboxes[i, 0:4] = new_x0, new_y1, new_x1, new_y0
    
        return bboxes
    
    def process_landmark(self, image, src_landmarks, tar_landmarks):
        
        image_size = image.shape
        src_tmp = [(int(xy[1]), int(xy[0])) for xy in src_landmarks]
        tar_tmp = [(int(xy[1]), int(xy[0])) for xy in tar_landmarks]
        M = umeyama(np.array(src_tmp), np.array(tar_tmp), True)[0:2]
        result = cv2.warpAffine(image, M, 
                                (image_size[1], image_size[0]), 
                                 borderMode=cv2.BORDER_REPLICATE)
        return result
    
    def process_landmark2(self, image, fa):
    
        is_face = False
        
        preds = fa.get_landmarks(image)
        if preds is not None:
            pred = preds[0]
            mask = np.zeros_like(image)
                    
            pnts_right = [(pred[i, 0], pred[i, 1]) for i in range(36, 42)]
            hull = cv2.convexHull(np.array(pnts_right)).astype(np.int32)
            mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)
                    
            pnts_left = [(pred[i, 0], pred[i, 1]) for i in range(42, 48)]
            hull = cv2.convexHull(np.array(pnts_left)).astype(np.int32)
            mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)
                    
            mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            is_face = True
        else:
            mask = np.zeros_like(image)
    
        return is_face, mask

    def process(self, input_image, save_path=None, save_type='A', 
                minsize=30, threshold=[0.6, 0.7, 0.7], factor=0.709):    
                
        boxes, pnts = face_detect.detect_face(input_image, minsize, 
                                              self.pnet, self.rnet, self.onet, threshold, factor)
        faces = self.process_bbox(boxes, input_image.shape)
        
        for idx, (x0, y1, x1, y0, conf_score) in enumerate(faces):
        
            det_face = input_image[int(x0):int(x1), int(y0):int(y1), :]
            
            src_landmarks = [(int(pnts[i+5][0]-x0), 
                              int(pnts[i][0]-y0)) for i in range(5)]
            image_size = (int(x1)-int(x0), int(y1)-int(y0))
            tar_landmarks = [(int(one[0]*image_size[0]), 
                              int(one[1]*image_size[1])) for one in self.ratio_landmarks]
            align_image = self.process_landmark(det_face, 
                                           src_landmarks, tar_landmarks)
            align_image = cv2.resize(align_image, (256, 256))
            
            if self.use_fa:
                is_face, mask_image = self.process_landmark2(align_image, self.fa)
            else:
                mask_image = np.zeros_like(align_image)
                mask_image[int(src_landmarks[0][0]-image_size[0]/15.0): 
                           int(src_landmarks[0][0]+image_size[0]/15.0),
                           int(src_landmarks[0][1]-image_size[1]/8.0):
                           int(src_landmarks[0][1]+image_size[1]/8.0), :] = 255
                mask_image[int(src_landmarks[1][0]-image_size[0]/15.0): 
                           int(src_landmarks[1][0]+image_size[0]/15.0),
                           int(src_landmarks[1][1]-image_size[1]/8.0): 
                           int(src_landmarks[1][1]+image_size[1]/8.0), :] = 255
                mask_image = self.process_landmark(mask_image, 
                                              src_landmarks, tar_landmarks)
            
            if is_face:
                self.image_label += 1
                cv2.imwrite(save_path + "/align_" + save_type + "/data_" + save_type + "_"
                            + str(self.image_label) + ".jpg", align_image)
                cv2.imwrite(save_path + "/mask_" + save_type + "/data_" + save_type + "_"
                            + str(self.image_label) + ".jpg", mask_image)
    
if __name__ == "__main__":
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    face_process = FaceProcess(use_fa=True)
    face_process.load_models()
    face_process.process(cv2.imread("test1.jpg"), "../Files", "A")   
    