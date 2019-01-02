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
                        help="Input File, such as '../Angelababy'.")
    parser.add_argument('output_dir', type=str, 
                        help="Directory, such as '../datasets/Angelababy'.")
    parser.add_argument('--save_type', type=str, 
                        help="Face Type, such as 'A' or 'B'.")

    return parser.parse_args(argv)
    
def main(args):
    
    # get params from system
    input_file = args.input_file
    output_dir = args.output_dir
    save_type = args.save_type

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
    for file_name in os.listdir(input_file):
        rfile_name = input_file + '/' + file_name
        for image_name in os.listdir(rfile_name):
            rimage_name = rfile_name + '/' + image_name
            try:
                face_process.process(cv2.imread(rimage_name), 
                                     output_dir, save_type)
            except Exception:
                pass
            counter += 1
            print("image name: %s, image counter: %d" %(rimage_name, counter))
            
    print("Done!")
        
if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    main(parse_arguments(sys.argv[1:]))
    