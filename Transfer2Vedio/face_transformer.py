import cv2
import numpy as np
import tensorflow as tf
from color_correction import adain
from color_correction import color_hist_match

class FaceTransformer(object):
    """
    Attributes:
        path_func: string, direction for the transformation: either AtoB or BtoA.
        model: the generator of the faceswap-GAN model
    """
    def __init__(self, image_shape=(64, 64, 3)): 
        self.path_func = None
        self.swap_sess_A = None
        self.swap_x_A = None
        self.swap_y_A = None
        self.swap_sess_B = None
        self.swap_x_B = None
        self.swap_y_B = None
        self.image_shape = image_shape
        
        self.inp_img = None
        self.input_size = None
        self.roi = None
        self.roi_size = None
        self.ae_input = None
        self.ae_output = None
        self.ae_output_masked = None
        self.ae_output_rgb = None
        self.ae_output_a = None
        self.result = None 
        self.result_rawRGB = None
        self.result_alpha = None
        
    def set_model(self, model_path):
        # face swap model A
        swap_graph_def = tf.GraphDef()
        swap_graph_def.ParseFromString(open(model_path + "/pico_FaceSwapA_model.pb", "rb").read())
        tf.import_graph_def(swap_graph_def, name="")
        self.swap_sess_A = tf.Session()
        self.swap_sess_A.graph.get_operations()
        self.swap_x_A = self.swap_sess_A.graph.get_tensor_by_name("swap_inputA:0")
        self.swap_y_A = self.swap_sess_A.graph.get_tensor_by_name("model_2/concatenate_1/concat:0")
        
        # face swap model B
        swap_graph_def = tf.GraphDef()
        swap_graph_def.ParseFromString(open(model_path + "/pico_FaceSwapB_model.pb", "rb").read())
        tf.import_graph_def(swap_graph_def, name="")
        self.swap_sess_B = tf.Session()
        self.swap_sess_B.graph.get_operations()
        self.swap_x_B = self.swap_sess_B.graph.get_tensor_by_name("swap_inputB:0")
        self.swap_y_B = self.swap_sess_B.graph.get_tensor_by_name("model_4/concatenate_2/concat:0")
    
    def _preprocess_inp_img(self, input_img, roi_coverage, IMAGE_SHAPE):
        input_size = input_img.shape        
        roi_x, roi_y = int(input_size[0]*(1-roi_coverage)), int(input_size[1]*(1-roi_coverage))
        roi = input_img[roi_x:-roi_x, roi_y:-roi_y,:] # RGB, [0, 255]  
        roi_size = roi.shape
        ae_input = cv2.resize(roi, IMAGE_SHAPE[:2]) / 255. * 2 - 1 # BGR, [-1, 1]  
        self.input_img = input_img
        self.input_size = input_size
        self.roi = roi
        self.roi_size = roi_size
        self.ae_input = ae_input
    
    def _ae_forward_pass(self, ae_input):
        ae_out = self.swap_sess_A
        self.ae_output = np.squeeze(np.array([ae_out]))        
        
    def _postprocess_roi_img(self, ae_output, roi, roi_size, color_correction):
        ae_output_a = ae_output[:,:,0] * 255.0
        ae_output_a = cv2.resize(ae_output_a, (roi_size[1],roi_size[0]))[...,np.newaxis]
        ae_output_rgb = np.clip( (ae_output[:,:,1:] + 1) * 255.0 / 2, 0, 255)
        ae_output_rgb = cv2.resize(ae_output_rgb, (roi_size[1],roi_size[0]))
        ae_output_masked = (ae_output_a/255.0 * ae_output_rgb + (1 - ae_output_a/255.0) * roi).astype('uint8')
        self.ae_output_a = ae_output_a         
        if color_correction == "adain":
            self.ae_output_masked = adain(ae_output_masked, roi)
            self.ae_output_rgb = adain(ae_output_rgb, roi)
        elif color_correction == "hist_match":
            self.ae_output_masked = color_hist_match(ae_output_masked, roi)
            self.ae_output_rgb = color_hist_match(ae_output_rgb, roi)
        else:
            self.ae_output_masked = ae_output_masked
            self.ae_output_rgb = ae_output_rgb
    
    def _merge_img_and_mask(self, ae_output_rgb, ae_output_masked, input_size, roi, roi_coverage):  
        blend_mask = self.get_feather_edges_mask(roi, roi_coverage)      
        blended_img = blend_mask/255.0 * ae_output_masked + (1-blend_mask/255.0) * roi
        result = self.inp_img.copy()
        roi_x, roi_y = int(input_size[0]*(1-roi_coverage)), int(input_size[1]*(1-roi_coverage))
        result[roi_x:-roi_x, roi_y:-roi_y,:] = blended_img.astype('uint8') 
        result_rawRGB = self.inp_img.copy()
        result_rawRGB[roi_x:-roi_x, roi_y:-roi_y,:] = ae_output_rgb 
        result_alpha = np.zeros_like(self.inp_img)
        result_alpha[roi_x:-roi_x, roi_y:-roi_y,:] = (blend_mask/255.0) * self.ae_output_a 
        self.result = result 
        self.result_rawRGB = result_rawRGB
        self.result_alpha = result_alpha
    
    @staticmethod
    def get_feather_edges_mask(img, roi_coverage):
        img_size = img.shape
        mask = np.zeros_like(img)
        roi_x, roi_y = int(img_size[0]*(1-roi_coverage)), int(img_size[1]*(1-roi_coverage))
        mask[roi_x:-roi_x, roi_y:-roi_y,:]  = 255
        mask = cv2.GaussianBlur(mask,(15,15),10)
        return mask  

    def transform(self, inp_img, direction, roi_coverage, color_correction):
        
        self.check_roi_coverage(inp_img, roi_coverage)
        
        self.inp_img = inp_img
        
        # pre-process input image
        # Set 5 members: self.inp_img, self.input_size, self.roi, self.roi_size, self.ae_input
        self._preprocess_inp_img(self.inp_img, roi_coverage, self.image_shape)

        # model inference
        # Set 1 member: self.ae_output
        if direction == "AtoB":
            ae_out = self.swap_sess_B.run(self.swap_y_B, feed_dict={self.swap_x_B: [self.ae_input]})
            self.ae_output = np.squeeze(np.array([ae_out]))
        elif direction == "BtoA":
            ae_out = self.swap_sess_A.run(self.swap_y_A, feed_dict={self.swap_x_A: [self.ae_input]})
            self.ae_output = np.squeeze(np.array([ae_out]))
        else:
            raise ValueError("direction should be either AtoB or BtoA, recieved {direction}.")

        # post-process transformed roi image
        # Set 3 members: self.ae_output_a, self.ae_output_masked, self.ae_output_bgr
        self._postprocess_roi_img(self.ae_output, self.roi, self.roi_size, color_correction)

        # merge transformed output back to input image
        # Set 3 members: self.result, self.result_rawRGB, self.result_alpha
        self._merge_img_and_mask(self.ae_output_rgb, self.ae_output_masked, 
                                 self.input_size, self.roi, roi_coverage)
        
        return self.result, self.result_rawRGB, self.result_alpha
    
    @staticmethod
    def check_roi_coverage(inp_img, roi_coverage):
        input_size = inp_img.shape        
        roi_x, roi_y = int(input_size[0]*(1-roi_coverage)), int(input_size[1]*(1-roi_coverage))
        if roi_x == 0 or roi_y == 0:
            raise ValueError("Error occurs when cropping roi image. \
            Consider increasing min_face_area or decreasing roi_coverage.")
        