# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:52:04 2018

@author: shen1994
"""

import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda
from keras.layers import concatenate
from tensorflow.contrib.distributions import Beta

from norm import InstanceNormalization

def first_order(x, axis=1):
    img_nrows = x.shape[1]
    img_ncols = x.shape[2]
    if axis == 1:
        return K.abs(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    elif axis == 2:
        return K.abs(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    else:
        return None
        
def cyclic_loss(netG1, netG2, real1):
    fake2 = netG2(real1)
    fake2 = Lambda(lambda x: x[:,:,:, 1:])(fake2)
    cyclic1 = netG1(fake2)
    cyclic1 = Lambda(lambda x: x[:,:,:, 1:])(cyclic1)
    loss =  K.mean(K.abs(cyclic1-real1))
    
    return loss

def define_loss(netD, netG, real, fake_argb, distorted, mask_eyes, mask, vggface_feats, 
                use_mixup, use_pl, use_hinge):   
    alpha = Lambda(lambda x: x[:,:,:, :1])(fake_argb)
    fake_rgb = Lambda(lambda x: x[:,:,:, 1:])(fake_argb)
    fake = alpha * fake_rgb + (1-alpha) * distorted

    loss_fn_l2 = lambda output, target : K.mean(K.abs(K.square(output-target)))
    loss_fn_l1 = lambda output, target : K.mean(K.abs(output-target))
    
    loss_D, loss_G = 0.0, 0.0
    
    if use_mixup:
        dist = Beta(0.2, 0.2)
        lam = dist.sample()
        
        mixup = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake, distorted])   
        output_fake = netD(concatenate([fake, distorted])) # dummy
        output_mixup = netD(mixup)
        loss_D += loss_fn_l2(output_mixup, lam * K.ones_like(output_mixup)) 
        loss_G += 0.1 * loss_fn_l2(output_fake, K.ones_like(output_fake))
        mixup2 = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake_rgb, distorted])
        output_fake_rgb = netD(concatenate([fake_rgb, distorted]))
        output_mixup2 = netD(mixup2)
        loss_D += loss_fn_l2(output_mixup2, lam * K.ones_like(output_mixup2))
        loss_G += 0.1 * loss_fn_l2(output_fake_rgb, K.ones_like(output_fake_rgb))
    else:
        output_real = netD(concatenate([real, distorted])) # positive sample
        output_fake = netD(concatenate([fake, distorted])) # negative sample   
        loss_D_real = loss_fn_l2(output_real, K.ones_like(output_real)) / 2.0   
        loss_D_fake = loss_fn_l2(output_fake, K.zeros_like(output_fake)) / 2.0    
        loss_D += loss_D_real + loss_D_fake
        loss_G += 0.1 * loss_fn_l2(output_fake, K.ones_like(output_fake))
        fake_output2 = netD(concatenate([fake_rgb, distorted]))
        loss_D += K.mean(K.square(output_real - K.mean(fake_output2,axis=0) - K.ones_like(fake_output2))) / 2.0 
        loss_D += K.mean(K.square(fake_output2 - K.mean(output_real,axis=0) - K.zeros_like(fake_output2))) / 2.0 
        loss_G += 0.1 * K.mean(K.square(output_real - K.mean(fake_output2,axis=0) - K.zeros_like(fake_output2))) / 2.0  
        loss_G += 0.1 * K.mean(K.square(fake_output2 - K.mean(output_real,axis=0) - K.ones_like(fake_output2))) / 2.0 

    # Reconstruction loss
    loss_G += 1.0 * loss_fn_l1(fake_rgb, real)
    loss_G += 35.0 * K.mean(K.abs(mask_eyes*(fake_rgb - real)))
    for out in netG.outputs[:-1]:
        out_size = out.get_shape().as_list()
        resized_real = tf.image.resize_images(real, out_size[1:3])
        loss_G += 1.0 * loss_fn_l1(out, resized_real)
    
    # Edge loss (similar with total variation loss) 
    loss_G += 0.15 * loss_fn_l1(first_order(fake_rgb, axis=1), first_order(real, axis=1))
    loss_G += 0.15 * loss_fn_l1(first_order(fake_rgb, axis=2), first_order(real, axis=2))
    shape_mask_eyes = mask_eyes.get_shape().as_list()
    resized_mask_eyes = tf.image.resize_images(mask_eyes, [shape_mask_eyes[1]-1, shape_mask_eyes[2]-1]) 
    loss_G += 35.0 * K.mean(K.abs(resized_mask_eyes * (first_order(fake_rgb, axis=1) - first_order(real, axis=1))))
    loss_G += 35.0 * K.mean(K.abs(resized_mask_eyes * (first_order(fake_rgb, axis=2) - first_order(real, axis=2)))) 
    
    # pl loss
    if use_pl:
        def preprocess_vggface(x):
            x = (x + 1)/2.0 * 255 # channel order: BGR
            x -= [91.4953, 103.8827, 131.0912]
            return x    
    
        real_sz224 = tf.image.resize_images(real, [224, 224])
        real_sz224 = Lambda(preprocess_vggface)(real_sz224)
        dist = Beta(0.2, 0.2)
        lam = dist.sample() # use mixup trick here to reduce foward pass from 2 times to 1.
        mixup = lam*fake_rgb + (1-lam)*fake
        fake_sz224 = tf.image.resize_images(mixup, [224, 224])
        fake_sz224 = Lambda(preprocess_vggface)(fake_sz224)
        real_feat112, real_feat55, real_feat28, real_feat7 = vggface_feats(real_sz224)
        fake_feat112, fake_feat55, fake_feat28, fake_feat7  = vggface_feats(fake_sz224)
        
        # Apply instance norm on VGG(ResNet) features
        # From MUNIT https://github.com/NVlabs/MUNIT
        def instnorm(): return InstanceNormalization()
        loss_G += 0.01 * loss_fn_l2(instnorm()(fake_feat7), instnorm()(real_feat7)) 
        loss_G += 0.1 * loss_fn_l2(instnorm()(fake_feat28), instnorm()(real_feat28))
        loss_G += 0.3 * loss_fn_l2(instnorm()(fake_feat55), instnorm()(real_feat55))
        loss_G += 0.1 * loss_fn_l2(instnorm()(fake_feat112), instnorm()(real_feat112))
    else:
        loss_G += K.zeros(1)
        
    # Alpha mask loss
    if use_hinge:
        loss_G += 1e-1 * K.mean(K.maximum(0., 0.1 - mask)) 
    else:
        loss_G += 1e-2 * K.mean(K.abs(mask))      
        
    # Alpha mask total variation loss
    loss_G += 0.01 * K.mean(first_order(mask, axis=1))
    loss_G += 0.01 * K.mean(first_order(mask, axis=2))
    
    return loss_D, loss_G

