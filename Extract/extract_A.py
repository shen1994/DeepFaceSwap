# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:44:05 2018

@author: shen1994
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import sys
import argparse
from pathlib import Path
from face_process import FaceProcess

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_file', type=str, 
                        help="Video File, such as '../Files/data_A.mp4'.")
    parser.add_argument('output_dir', type=str, 
                        help="Directory, such as '../Files'.")
    parser.add_argument('--save_type', type=str, 
                        help="Face Type, such as 'A' or 'B'.")
    parser.add_argument('--interval', type=int, default=1, 
                        help="Time Interval, such as 1, 2, 3...")

    return parser.parse_args(argv)
    
def main(args):
    
    # get params from system
    input_file = args.input_file
    output_dir = args.output_dir
    save_type = args.save_type
    interval = args.interval

    # build paths
    dir_faceA = output_dir + "/" + save_type
    dir_faceA_align = output_dir + "/align_" + save_type
    dir_faceA_mask = output_dir + "/mask_" + save_type
    Path("%s"%dir_faceA).mkdir(parents=True, exist_ok=True)
    Path("%s"%dir_faceA_align).mkdir(parents=True, exist_ok=True)   
    Path("%s"%dir_faceA_mask).mkdir(parents=True, exist_ok=True) 
    
    # load models
    face_process = FaceProcess(use_fa=True)
    face_process.load_models()
        
    # extract aligned images and mask images
    counter = 0
    v_cap = cv2.VideoCapture(input_file)
    fps = v_cap.get(cv2.CAP_PROP_FPS)
    fnums = v_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    success, frame = v_cap.read()
    while success :
        if counter % interval == 0:
            face_process.process(frame, output_dir, save_type)
        counter += 1
        print("all frames: %d, one frame: %d" %(fnums, counter))
        cv2.waitKey(1000//int(fps))
        success, frame = v_cap.read()
    v_cap.release()

    # process has been down
    print("Done!")  
    
        
if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    main(parse_arguments(sys.argv[1:]))
    