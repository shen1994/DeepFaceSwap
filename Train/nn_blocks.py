# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 14:32:19 2018

@author: shen1994
"""

import tensorflow as tf
from keras import backend as K
from keras import initializers
from keras.layers.core import Layer
from keras.engine import InputSpec
from keras.layers import regularizers
from keras.layers import Reshape
from keras.layers import Softmax
from keras.layers import Lambda
from keras.layers import add
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
from pixel_shuffler import PixelShuffler
from norm import InstanceNormalization

conv_init = RandomNormal(0, 0.02)

class Scale(Layer):
    '''
    Code borrows from https://github.com/flyyufelix/cnn_finetune
    '''
    def __init__(self, weights=None, axis=-1, gamma_init='zero', **kwargs):
        self.axis = axis
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

        # Compatibility with TensorFlow >= 1.0.0
        self.gamma = K.variable(self.gamma_init((1,)), name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        return self.gamma * x

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def self_attn_block(inp, nc, squeeze_factor=8, w_l2=1e-4):
    '''
    Code borrows from https://github.com/taki0112/Self-Attention-GAN-Tensorflow
    '''
    x = inp
    shape_x = x.get_shape().as_list()
    
    f = Conv2D(nc//squeeze_factor, 1, kernel_regularizer=regularizers.l2(w_l2))(x)
    g = Conv2D(nc//squeeze_factor, 1, kernel_regularizer=regularizers.l2(w_l2))(x)
    h = Conv2D(nc, 1, kernel_regularizer=regularizers.l2(w_l2))(x)
    
    shape_f = f.get_shape().as_list()
    shape_g = g.get_shape().as_list()
    shape_h = h.get_shape().as_list()
    flat_f = Reshape((-1, shape_f[-1]))(f)
    flat_g = Reshape((-1, shape_g[-1]))(g)
    flat_h = Reshape((-1, shape_h[-1]))(h)   
    
    s = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([flat_g, flat_f])

    beta = Softmax(axis=-1)(s)
    o = Lambda(lambda x: tf.matmul(x[0], x[1]))([beta, flat_h])
    o = Reshape(shape_x[1:])(o)
    o = Scale()(o)
    
    out = add([o, inp])
    
    return out
    
def conv_block(input_tensor, f, use_norm=False, strides=2, w_l2=1e-4):
    x = input_tensor
    x = Conv2D(f, kernel_size=3, strides=strides, kernel_regularizer=regularizers.l2(w_l2),  
               kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = Activation("relu")(x)
    x = InstanceNormalization()(x) if use_norm else x
    return x

def conv_block_d(input_tensor, f, use_instance_norm=False, w_l2=1e-4):
    x = input_tensor
    x = Conv2D(f, kernel_size=4, strides=2, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def res_block(input_tensor, f, use_norm=False, w_l2=1e-4):
    x = input_tensor
    x = Conv2D(f, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(f, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = add([x, input_tensor])
    x = LeakyReLU(alpha=0.2)(x)
    x = InstanceNormalization()(x) if use_norm else x
    return x
    
def upscale_ps(input_tensor, filters, use_norm=False, w_l2=1e-4):
    x = input_tensor
    x = Conv2D(filters*4, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=RandomNormal(0, 0.02), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = InstanceNormalization()(x) if use_norm else x
    x = PixelShuffler()(x)
    
    return x

    