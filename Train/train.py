# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:38:35 2018

@author: shen1994
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import argparse
from keras.layers import Input
from keras.models import Model
from keras_vggface.vggface import VGGFace

from model import Encoder
from model import Decoder
from model import Discriminator
from generate import Generator
from build import build_model
from build import build_pl_model
from show import showG
from show import showG_mask

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('A_file', type=str, 
                        help="A File, such as '../Files/People'.")
    parser.add_argument('B_file', type=str, 
                        help="B File, such as '../Files/Angelababy'.")
    parser.add_argument('--GPU_number', type=int, default=0, 
                        help="Choose one GPU, such as 0, 1, 2...")
    parser.add_argument('--batch_size', type=int, default=12, 
                        help="Feed batch size, such as 8, 16, 32...")    

    return parser.parse_args(argv)

def main(args):
    
    # get params from system
    A_file = args.A_file
    B_file = args.B_file
    GPU_number = args.GPU_number
    batch_size = args.batch_size
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_number)

    if not os.path.exists("../Files/Model"):
        os.mkdir("../Files/Model")
    
    epochs = 1000000
    batch_size = batch_size
    image_shape = (128, 128, 3)
    gan_shape = (128, 128, 6)

    # define models
    x = Input(shape=image_shape, name='swap_input')
    encoder = Encoder(shape=image_shape)
    decoder_A = Decoder(shape=image_shape)
    decoder_B = Decoder(shape=image_shape)
    netGA = Model(x, decoder_A(encoder(x)))
    netGB = Model(x, decoder_B(encoder(x)))
    netDA = Discriminator(shape=gan_shape)
    netDB = Discriminator(shape=gan_shape)

    # load weights
    try:
        encoder  .load_weights( "../Files/Model/netEN.h5" )
        decoder_A.load_weights( "../Files/Model/netGA.h5" )
        decoder_B.load_weights( "../Files/Model/netGB.h5" )
        netDA.load_weights("../Files/Model/netDA.h5")
        netDB.load_weights("../Files/Model/netDB.h5")
    except:
        pass
    
    # if python > 2.0 use require_flatten instead of include_top
    vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
    vggface_feats = build_pl_model(vggface_model=vggface, before_activ=False)
    
    netDA_train, netGA_train, netDB_train, netGB_train, \
    path_A, path_B, path_mask_A, path_mask_B = \
        build_model(netDA, netGA, netDB, netGB, image_shape, vggface_feats, 
                    use_mixup=True, use_pl=True, use_hinge=False, use_cyclic=False)

    A_generate = Generator(image_path=A_file + "/align_A", mask_path=A_file + "/mask_A", 
                           batch_size=batch_size,
                           image_shape=image_shape).generate()
    B_generate = Generator(image_path=B_file + "/align_B", mask_path=B_file + "/mask_B", 
                           batch_size=batch_size,
                           image_shape=image_shape).generate()

    err_DA = err_GA = err_DB = err_GB = 0
    
    for epoch in range(epochs):
        
        warp_A, targ_A, mask_A = A_generate.__next__()
        warp_B, targ_B, mask_B = B_generate.__next__()
        coss_DA = netDA_train([warp_A, targ_A]) 
        coss_DB = netDB_train([warp_B, targ_B])

        warp_A, targ_A, mask_A = A_generate.__next__()
        warp_B, targ_B, mask_B = B_generate.__next__()        
        coss_GA = netGA_train([warp_A, targ_A, mask_A])
        coss_GB = netGB_train([warp_B, targ_B, mask_B])
        
        err_DA += coss_DA[0]
        err_GA += coss_GA[0]
        err_DB += coss_DB[0]
        err_GB += coss_GB[0]

        if epoch % 100 == 0:

            encoder  .save_weights("../Files/Model/netEN.h5")
            decoder_A.save_weights("../Files/Model/netGA.h5")
            decoder_B.save_weights("../Files/Model/netGB.h5")
            netDA.save_weights("../Files/Model/netDA.h5")
            netDB.save_weights("../Files/Model/netDB.h5")
  
            showG(warp_A, warp_B, path_A, path_B, 
                  batch_size=batch_size, name='rgb' + str(GPU_number))         
            showG_mask(targ_A, warp_B, path_mask_A, path_mask_B, 
                       batch_size=batch_size, name='mask' + str(GPU_number)) 
            
            print("E: %d---LOSS DA: %.2f, LOSS GA: %.2f, LOSS DB: %.2f, LOSS GB %.2f" %(epoch, err_DA, err_GA, err_DB, err_GB))
            
            err_DA = err_GA = err_DB = err_GB = 0

        if cv2.waitKey(1) == ord('q'):
            encoder  .save_weights("../Files/Model/netEN.h5")
            decoder_A.save_weights("../Files/Model/netGA.h5")
            decoder_B.save_weights("../Files/Model/netGB.h5")
            netDA.save_weights("../Files/Model/netDA.h5")
            netDB.save_weights("../Files/Model/netDB.h5")
            exit()
    
if __name__ == "__main__":    
    
    main(parse_arguments(sys.argv[1:]))
    