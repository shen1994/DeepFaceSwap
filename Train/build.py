# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:46:37 2018

@author: shen1994
"""

from keras import backend as K
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import concatenate
from keras.models import Model
from keras.optimizers import Adam
from losses import define_loss
from losses import cyclic_loss

def cycle_variables(netG):
    
    distorted_input = netG.inputs[0]
    fake_output = netG.outputs[-1]
    alpha = Lambda(lambda x: x[:,:,:, :1])(fake_output)
    rgb = Lambda(lambda x: x[:,:,:, 1:])(fake_output)
    
    masked_fake_output = alpha * rgb + (1-alpha) * distorted_input 
    
    fn_generate = K.function([distorted_input], [masked_fake_output])
    fn_mask = K.function([distorted_input], [concatenate([alpha, alpha, alpha])])
    fn_argb = K.function([distorted_input], [concatenate([alpha, rgb])])
    
    return distorted_input, fake_output, alpha, fn_generate, fn_mask, fn_argb 
    
def build_pl_model(vggface_model, before_activ=False):
    
    # Define Perceptual Loss Model
    vggface_model.trainable = False
    if before_activ == False:
        out_size112 = vggface_model.layers[1].output
        out_size55 = vggface_model.layers[36].output
        out_size28 = vggface_model.layers[78].output
        out_size7 = vggface_model.layers[-2].output
    else:
        out_size112 = vggface_model.layers[15].output
        out_size55 = vggface_model.layers[35].output
        out_size28 = vggface_model.layers[77].output
        out_size7 = vggface_model.layers[-3].output

    vggface_feats = Model(vggface_model.input, [out_size112, out_size55, out_size28, out_size7])
    vggface_feats.trainable = False
    
    return vggface_feats

def build_model(netDA, netGA, netDB, netGB, image_shape, vggface_feats, 
                use_mixup=True, use_pl=False, use_hinge=False, use_cyclic=True):
    
    # define varibles
    distorted_A, fake_A, mask_A, path_A, path_mask_A, path_abgr_A = cycle_variables(netGA)
    distorted_B, fake_B, mask_B, path_B, path_mask_B, path_abgr_B = cycle_variables(netGB)
    real_A = Input(shape=image_shape)
    real_B = Input(shape=image_shape)
    mask_eyes_A = Input(shape=image_shape)
    mask_eyes_B = Input(shape=image_shape)
    
    # define loss
    loss_DA, loss_GA = define_loss(netDA, netGA, real_A, fake_A, distorted_A, mask_eyes_A, mask_A, vggface_feats,
                                   use_mixup, use_pl, use_hinge)
    loss_DB, loss_GB = define_loss(netDB, netGB, real_B, fake_B, distorted_B, mask_eyes_B, mask_B, vggface_feats,
                                   use_mixup, use_pl, use_hinge)
    
    if use_cyclic:
        loss_GA += 10.0 * cyclic_loss(netGA, netGB, real_A)
        loss_GB += 10.0 * cyclic_loss(netGB, netGA, real_B)
    
    # L2 weight decay
    # https://github.com/keras-team/keras/issues/2662
    for loss_tensor in netGA.losses:
        loss_GA += loss_tensor
    for loss_tensor in netGB.losses:
        loss_GB += loss_tensor
    for loss_tensor in netDA.losses:
        loss_DA += loss_tensor
    for loss_tensor in netDB.losses:
        loss_DB += loss_tensor
        
    weightsDA = netDA.trainable_weights
    weightsGA = netGA.trainable_weights
    weightsDB = netDB.trainable_weights
    weightsGB = netGB.trainable_weights
    
    lrD = 2e-4
    lrG = 1e-4
    lr_factor = 1.0 # 1.0 -> 0.3
    training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDA,[],loss_DA)
    netDA_train = K.function([distorted_A, real_A],[loss_DA], training_updates)
    training_updates = Adam(lr=lrG*lr_factor, beta_1=0.5).get_updates(weightsGA,[], loss_GA)
    netGA_train = K.function([distorted_A, real_A, mask_eyes_A], [loss_GA], training_updates)

    training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDB,[],loss_DB)
    netDB_train = K.function([distorted_B, real_B],[loss_DB], training_updates)
    training_updates = Adam(lr=lrG*lr_factor, beta_1=0.5).get_updates(weightsGB,[], loss_GB)
    netGB_train = K.function([distorted_B, real_B, mask_eyes_B], [loss_GB], training_updates)  
    
    return netDA_train, netGA_train, netDB_train, netGB_train, \
                path_A, path_B, path_mask_A, path_mask_B
                
                
    