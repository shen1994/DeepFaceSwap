# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 17:10:34 2018

@author: shen1994
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 16:11:17 2018

@author: shen1994
"""

import os
import cv2
import numpy as np
from umeyama import umeyama

class Generator(object):
    
    def __init__(self,
             image_path=None,
             mask_path=None,
             batch_size=8,
             image_shape=(64, 64, 3),
             is_enhance=True):
        self.image_path = image_path
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.is_enhance = is_enhance

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])
        
    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * 0.5 + 0.5 
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * 0.5 + 0.5
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * 0.5 + 0.5
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * 0.5
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)
        
    def random_transform(self, image, mask, rotation_range, zoom_range, shift_range, random_flip):
        h,w = image.shape[0:2]
        rotation = np.random.uniform( -rotation_range, rotation_range )
        scale = np.random.uniform( 1 - zoom_range, 1 + zoom_range )
        tx = np.random.uniform( -shift_range, shift_range ) * w
        ty = np.random.uniform( -shift_range, shift_range ) * h
        mat = cv2.getRotationMatrix2D( (w//2,h//2), rotation, scale )
        mat[:,2] += (tx,ty)
        image_result = cv2.warpAffine( image, mat, (w,h), borderMode=cv2.BORDER_REPLICATE )
        mask_result = cv2.warpAffine( mask, mat, (w,h), borderMode=cv2.BORDER_REPLICATE )
        if np.random.random() < random_flip:
            image_result = image_result[:,::-1]
            mask_result = mask_result[:,::-1]
        
        return image_result, mask_result
        
    def random_warp(self, image, mask, res=64):
        assert image.shape == (256,256,3)
        res_scale = res//64
        interp_param = 80 * res_scale
        interp_slice = slice(interp_param//10,9*interp_param//10)
        dst_pnts_slice = slice(0,65*res_scale,16*res_scale)
        
        rand_coverage = np.random.randint(20) + 85 # random warping coverage
        rand_scale = np.random.uniform(5., 6.2) # random warping scale
        
        range_ = np.linspace(128-rand_coverage, 128+rand_coverage, 5)
        mapx = np.broadcast_to(range_, (5,5))
        mapy = mapx.T
        mapx = mapx + np.random.normal(size=(5,5), scale=rand_scale)
        mapy = mapy + np.random.normal(size=(5,5), scale=rand_scale)
        interp_mapx = cv2.resize(mapx, (interp_param,interp_param))[interp_slice,interp_slice].astype('float32')
        interp_mapy = cv2.resize(mapy, (interp_param,interp_param))[interp_slice,interp_slice].astype('float32')
        warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
        if self.is_enhance:
            warped_image = self.saturation(warped_image)
            warped_image = self.brightness(warped_image)
            warped_image = self.contrast(warped_image)
            warped_image = self.lighting(warped_image)
        src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
        dst_points = np.mgrid[dst_pnts_slice,dst_pnts_slice].T.reshape(-1,2)
        mat = umeyama(src_points, dst_points, True)[0:2]
        target_image = cv2.warpAffine(image, mat, (res,res))
        mask_image = cv2.warpAffine(mask, mat, (res,res))
        return warped_image, target_image, mask_image
        
    def generate(self):
        
        while(True):
            
            image_samples = 0
            
            dir_images, dir_masks = [], []
            for image_name in os.listdir(self.image_path):
                dir_images.append(self.image_path + os.sep + image_name)
                dir_masks.append(self.mask_path + os.sep + image_name)
                image_samples += 1
            
            rand_indexes = [i for i in range(image_samples)]
            np.random.shuffle(rand_indexes)
            dir_images = [dir_images[one] for one in rand_indexes]
            dir_masks = [dir_masks[one] for one in rand_indexes]
            
            counter = 0
            images_warps, images_targs, images_masks = [], [], []
            for i in range(image_samples):
                image = cv2.imread(dir_images[i])
                image = cv2.resize(image, (256, 256))
                mask = cv2.imread(dir_masks[i])
                mask = cv2.resize(mask, (256, 256))
                image, mask = self.random_transform(image, mask, 15, 0.1, 0.05, 0.5)
                warp_image, targ_image, mask_image = self.random_warp(image, mask, 
                                                                      res=self.image_shape[0])
                warp_image = warp_image / 255.0 * 2 - 1
                targ_image = targ_image / 255.0 * 2 - 1
                mask_image = mask_image / 255.0 * 2 - 1
                
                images_warps.append(warp_image)
                images_targs.append(targ_image)
                images_masks.append(mask_image)
                counter += 1
                if counter == self.batch_size:
                    yield (np.array(images_warps), np.array(images_targs), np.array(images_masks))
                    counter = 0
                    images_warps, images_targs, images_masks = [], [], []  
    
    
            