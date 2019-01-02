# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:23:23 2018

@author: shen1994
"""

from keras.models import Model
from keras.initializers import RandomNormal
from keras.layers import Input, Conv2D, Dense, Flatten, Reshape, concatenate

from nn_blocks import conv_block
from nn_blocks import self_attn_block
from nn_blocks import upscale_ps
from nn_blocks import res_block
from nn_blocks import conv_block_d

conv_init = RandomNormal(0, 0.02)

def Encoder(shape=(64, 64, 3)):

    x = Input(shape=shape, name='encoder') 
    x_c = Conv2D(64, kernel_size=5, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x_c = conv_block(x_c, 128)   
    x_c = conv_block(x_c, 256, use_norm=True)   
    x_c = self_attn_block(x_c, 256)    
    x_c = conv_block(x_c, 512, use_norm=True)    
    x_c = self_attn_block(x_c, 512)
    x_c = conv_block(x_c, 1024, use_norm=True)
    
    activ_map_size = shape[0] // 16
    while(activ_map_size > 4):
        x_c = conv_block(x_c, 1024, use_norm=True) 
        activ_map_size = activ_map_size // 2
    
    x_c = Dense(1024)(Flatten()(x_c))
    x_c = Dense(4*4*1024)(x_c)
    x_c = Reshape((4,4,1024))(x_c)    
    y = upscale_ps(x_c, 512, use_norm=True)

    return Model(x, y)
    
def Decoder(shape=(64, 64, 3)):

    x = Input(shape=(8, 8, 512), name='decoder')
    x_c = upscale_ps(x, 256, use_norm=True)  
    x_c = upscale_ps(x_c, 128, use_norm=True)    
    x_c = self_attn_block(x_c, 128)    
    x_c = upscale_ps(x_c, 64, use_norm=True)  
    x_c = res_block(x_c, 64, use_norm=True)
    x_c = self_attn_block(x_c, 64)

    activ_map_size = 64 # 8 * 8
    while (activ_map_size < shape[0]):
        x_c = upscale_ps(x_c, 64, use_norm=True)
        x_c = conv_block(x_c, 64, strides=1)
        activ_map_size *= 2
    
    alpha = Conv2D(1, kernel_size=5, kernel_initializer=conv_init, padding='same', activation="sigmoid")(x_c)
    rgb = Conv2D(3, kernel_size=5, kernel_initializer=conv_init, padding='same', activation="tanh")(x_c)
    out = concatenate([alpha, rgb])
    
    return Model(x, [out])

def Discriminator(shape=(64, 64, 6)):
    
    x = Input(shape=shape, name='discriminator')
    x_c = conv_block_d(x, 64, False)
    x_c = conv_block_d(x_c, 128, True)
    x_c = conv_block_d(x_c, 256, True)
    x_c = self_attn_block(x_c, 256)
    
    activ_map_size = shape[0] // 8
    while(activ_map_size > 8):
        x_c = conv_block_d(x_c, 256, True)
        x_c = self_attn_block(x_c, 256)
        activ_map_size = activ_map_size // 2
        
    y = Conv2D(1, kernel_size=4, kernel_initializer=conv_init, use_bias=False, padding="same")(x_c)   
    
    return Model(inputs=[x], outputs=y) 
    
    
    
    
    
    
    
    
    
    
    
    