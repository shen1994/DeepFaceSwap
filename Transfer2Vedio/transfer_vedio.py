# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 09:35:12 2018

@author: shen1994
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import argparse
import face_detect
import numpy as np
import tensorflow as tf
from umeyama import umeyama
from face_transformer import FaceTransformer

def process_bbox(bboxes, image_shape):
        
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

def process_landmark(image, src_landmarks, tar_landmarks):
        
    image_size = image.shape
    src_tmp = [(int(xy[1]), int(xy[0])) for xy in src_landmarks]
    tar_tmp = [(int(xy[1]), int(xy[0])) for xy in tar_landmarks]
    M = umeyama(np.array(src_tmp), np.array(tar_tmp), True)[0:2]
    result = cv2.warpAffine(image, M, 
                            (image_size[1], image_size[0]), 
                            borderMode=cv2.BORDER_REPLICATE)
    return result
    
def process_one_image(image, pnet, rnet, onet, transfer_model, best_score, ratio_landmarks,
                      direction="AtoB", roi_coverage=0.90, color_correction="hist_match", 
                      minsize=20, threshold=[0.6, 0.7, 0.7], factor=0.709):
    
    comb_image = image.copy()
    
    faces, pnts = face_detect.detect_face(image, minsize, 
                                          pnet, rnet, onet, threshold, factor)
    if len(faces) == 0:
        return False, image
        
    faces = process_bbox(faces, image.shape)
    
    for i, (x0, y1, x1, y0, conf_score) in enumerate(faces):

        if conf_score > best_score:
            
            det_image = image[int(x0):int(x1), int(y0):int(y1), :]

            src_landmarks = [(int(pnts[i+5][0]-x0), 
                              int(pnts[i][0]-y0)) for i in range(5)]
            image_size = (int(x1)-int(x0), int(y1)-int(y0))
            tar_landmarks = [(int(one[0]*image_size[0]), 
                              int(one[1]*image_size[1])) for one in ratio_landmarks]
                                  
            align_image = process_landmark(det_image, 
                                           src_landmarks, tar_landmarks)
            
            r_im, r_rgb, r_a = transfer_model.transform(align_image, direction=direction,
                                                roi_coverage=roi_coverage,
                                                color_correction=color_correction)
            rev_det_im = process_landmark(r_im, 
                                          tar_landmarks, src_landmarks)
            rev_ali_ma = process_landmark(r_a,
                                          tar_landmarks, src_landmarks)
            rev_ali_pr =  rev_ali_ma / 255.0
            result = np.zeros_like(det_image)
            result = (rev_ali_pr * rev_det_im + (1 - rev_ali_pr) * det_image).astype('uint8') 
            
            cv2.imwrite("test.jpg", result)
            cv2.imwrite("det.jpg", det_image)
    
            comb_image[int(x0):int(x1), int(y0):int(y1), :] = result

    return True, comb_image   
    
def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_file', type=str, 
                        help="To Transfer File, such as '../Files/LiaoFan/data_A.mp4'.")
    parser.add_argument('generated_file', type=str, 
                        help="Generated File, such as '../Files/Transfer/LiaoFan2XiaoShen.mp4'.")
    parser.add_argument('model_path', type=str, 
                        help="Model Path, such as '../Files/Model'.")
    parser.add_argument('--GPU_number', type=int, default=0, 
                        help="Choose one GPU, such as 0, 1, 2...")
    parser.add_argument('--direction', type=str, default='AtoB', 
                        help="Transfer Direction, such as 'AtoB' or 'BtoA'")

    return parser.parse_args(argv)

def main(args):
    
    # get params from system
    input_file = args.input_file
    generated_file = args.generated_file
    model_path = args.model_path
    GPU_number = args.GPU_number
    direction = args.direction
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_number)
    
    # create folders
    i_files = input_file.split('/')
    i_file = ''
    for index in range(len(i_files) - 1):
        i_file += i_files[index]
        i_file += '/'
    if not os.path.exists(i_file + 'swap'):
        os.mkdir(i_file + 'swap')

    g_files = generated_file.split('/')
    g_file = ''
    for index in range(len(g_files) - 1):
        g_file += g_files[index]
        g_file += '/'
    g_file = g_file[:-1]
    if not os.path.exists(g_file):
        os.mkdir(g_file)

    best_score = 0.70
    ratio_landmarks = [
        (0.31339227236234224, 0.3259269274198092),
        (0.31075140146108776, 0.7228453709528997),
        (0.5523683107816256, 0.5187296867370605),
        (0.7752419985257663, 0.37262483743520886),
        (0.7759613623985877, 0.6772957581740159)] 
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = face_detect.create_mtcnn(sess, None)

    transfer_model = FaceTransformer(image_shape=(128, 128, 3))
    transfer_model.set_model(model_path)
          
    v_cap = cv2.VideoCapture(input_file)
    
    fps = v_cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(v_cap.get(cv2.CAP_PROP_FOURCC))
    size = (int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter_A = cv2.VideoWriter(generated_file, fourcc, fps, size, True)
    videoWriter_B = cv2.VideoWriter('data.mp4', fourcc, fps, size, True)

    counter = 0
    save_counter = 0    
    success, frame = v_cap.read()
    while success :
        try:
            '''
            if counter < 60:
                is_face, swap_image = process_one_image(frame, pnet, rnet, onet, transfer_model, 
                                                        best_score, ratio_landmarks, direction=direction, roi_coverage=0.88)     
            elif counter < 100:
                is_face, swap_image = process_one_image(frame, pnet, rnet, onet, transfer_model, 
                                                        best_score, ratio_landmarks, direction=direction, roi_coverage=0.91)  
            elif counter == 107 or counter == 109 or counter == 112 or counter == 122 or counter == 351:
                pass
            elif counter < 400:
                is_face, swap_image = process_one_image(frame, pnet, rnet, onet, transfer_model, 
                                                        best_score, ratio_landmarks, direction=direction, roi_coverage=0.88) 
            elif counter == 418 and counter == 419 or counter == 420 or counter == 440 or counter == 457 \
                or counter == 458 or counter == 465 or counter == 499:
                pass
            elif counter < 500:
                is_face, swap_image = process_one_image(frame, pnet, rnet, onet, transfer_model, 
                                                        best_score, ratio_landmarks, direction=direction, roi_coverage=0.95)
            elif counter == 502 or counter == 507 or counter == 508 or counter == 512 or counter == 523 or counter == 524 \
                or counter == 525 or counter == 546 or counter == 594 or counter == 606 or counter == 607 or counter == 622 \
                or counter == 661 or counter == 670 or counter == 671 or counter == 672 or counter == 700 or counter == 702 \
                or counter == 727 or counter == 736 or counter == 794 or counter == 796 or counter == 893 or counter == 1315:
                pass
            else:
                is_face, swap_image = process_one_image(frame, pnet, rnet, onet, transfer_model, 
                                                        best_score, ratio_landmarks, direction=direction, roi_coverage=0.895)

            if counter <= 530
                pass
            elif counter >= 1450:
                videoWriter_A.write(frame)
                videoWriter_B.write(frame)
            else:
                if is_face:
                    cv2.imwrite(i_file+'swap/'+str(counter)+'.jpg', swap_image)
                    videoWriter_A.write(swap_image)
                    videoWriter_B.write(frame)
            '''
            if counter < 500:
                is_face, swap_image = process_one_image(frame, pnet, rnet, onet, transfer_model, 
                                                        best_score, ratio_landmarks, direction=direction, roi_coverage=0.885)
            else:
                is_face, swap_image = process_one_image(frame, pnet, rnet, onet, transfer_model, 
                                                        best_score, ratio_landmarks, direction=direction, roi_coverage=0.84)                
            counter_list = [28, 46, 237, 239, 288, 289, 328, 329, 337, 339, 361, 362, 455, 457, 460, 480, 481, 
                            581, 582, 583, 584, 585, 586, 588, 589, 590, 591, 592, 593, 594, 596, 599, 600,
                            601, 603, 609, 613, 614, 615, 616, 620, 622, 623, 624, 633, 634, 635, 636, 638, 639, 640, 641, 643, 644,
                            647, 648, 651, 661, 678, 679, 683, 685, 686, 689, 692, 696, 698, 699, 700, 701, 711, 712, 713, 
                            721, 744, 745, 757, 758, 236, 441, 465, 466, 478, 485, 481, 492, 500, 508, 509, 511, 523, 525, 531, 532,
                            533, 534, 535, 536, 551, 552, 564, 569, 574, 578, 579, 491]
            if counter in counter_list:
                if not save_counter == 0:
                    videoWriter_A.write(cv2.imread("..Files/Deve/swap/"+str(save_counter)+".jpg", 1))
                    videoWriter_B.write(frame)
            elif counter < 718:
                if is_face:
                    cv2.imwrite(i_file+'swap/'+str(counter)+'.jpg', swap_image)
                    videoWriter_A.write(swap_image)
                    videoWriter_B.write(frame)
                    cv2.imshow("dd", swap_image)
                    save_counter = counter
            else:
                # cv2.imwrite(i_file+'swap/'+str(counter)+'.jpg', swap_image)
                videoWriter_A.write(frame)
                videoWriter_B.write(frame)   
                cv2.imshow("dd", frame)
        
                    
        except Exception:
            pass
        counter += 1
        print('PROCESSING %d OK' %(counter))
        cv2.waitKey(1000//int(fps))
        success, frame = v_cap.read()
        
    print('PROCESSING IS OK!')
    v_cap.release()
    videoWriter_A.release()
    videoWriter_B.release()

if __name__ == "__main__":    
    
    main(parse_arguments(sys.argv[1:]))
    