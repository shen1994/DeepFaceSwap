# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 09:18:01 2018

@author: shen1994
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from model import Encoder
from model import Decoder
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.framework import graph_io

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a prunned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    prunned so subgraphs that are not neccesary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph
        
def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model_path', type=str, 
                        help="Generated Model Path, such as '../Files/Model'.")
    parser.add_argument('--GPU_number', type=int, default=0, 
                        help="Choose one GPU, such as 0, 1, 2...")

    return parser.parse_args(argv)

def main(args):

    model_path = args.model_path
    GPU_number = args.GPU_number
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_number)

    image_shape=(128, 128, 3)
       
    encoder = Encoder(shape=image_shape)
    encoder  .load_weights(model_path + "/netEN.h5")

    x_A = Input(shape=image_shape, name='swap_inputA') 
    decoder_A = Decoder(shape=image_shape)
    decoder_A.load_weights(model_path + "/netGA.h5")
    netGA = Model(x_A, decoder_A(encoder(x_A)))
    print('netGA input, output name is: %s, %s' %(netGA.input.name, netGA.output.name))
    frozen_graph = freeze_session(K.get_session(), output_names=[netGA.output.op.name])
    graph_io.write_graph(frozen_graph, model_path, "pico_FaceSwapA_model.pb", as_text=False)

    x_B = Input(shape=image_shape, name='swap_inputB')
    decoder_B = Decoder(shape=image_shape)
    decoder_B.load_weights(model_path + "/netGB.h5")
    netGB = Model(x_B, decoder_B(encoder(x_B)))
    print('netGB input, output name is: %s, %s' %(netGB.input.name, netGB.output.name))
    frozen_graph = freeze_session(K.get_session(), output_names=[netGB.output.op.name])
    graph_io.write_graph(frozen_graph, model_path, "pico_FaceSwapB_model.pb", as_text=False)
    
if __name__ == "__main__":    
    
    main(parse_arguments(sys.argv[1:]))   
    