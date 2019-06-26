import torch
import torch.nn.functional as F
from embiggen import *
import cv2
from PIL import Image
import random

import torchvision.transforms.functional as TF
###IMAGE PROCESSING TOOLS######
def upscaling_tensor_from_scene(scene, upsamling_size = [384,384], upsamling_mode = 'bicubic' ,  align_corner = True):
    lr, sm = get_lr_sm_from_scene(scene)
    bicubic_lr = F.interpolate(torch.tensor(lr), size = upsamling_size, mode= upsamling_mode, align_corners= align_corner) 
    bicubic_sm = F.interpolate(torch.tensor(sm), size = upsamling_size, mode= upsamling_mode, align_corners= align_corner)
    return bicubic_lr, bicubic_sm
def get_lr_sm_from_scene(scene):
    lrsm = list(lowres_image_iterator(scene))
    lr, sm= np.split(np.array(lrsm),2, axis = 1)    
    return lr, sm 
def mask_filter(bicubic_lr, bicubic_sm):
    filteret = bicubic_lr*bicubic_sm
    return filteret
def get_mean_tensor_with_sm(bicubic_lr, bicubic_sm):
    mean = torch.div(torch.sum(bicubic_lr*bicubic_sm, dim = 0), torch.sum(bicubic_sm, dim = 0))
    return mean
def np_3d_to_PIL(inputs):    #convert a 3d numpy array to PIL
    outputs = [Image.fromarray(inputs[i]) for i in range(len(inputs))] 
    return outputs
def PIL_3d_to_np(inputs):  #convert an array of PIL array to numpy
    outputs = np.array([np.array(inputs[i]) for i in range(len(inputs))])
    return outputs
def rotate_lr_hr(pic1, pic2):
    if random.random() > 5:
        angle = random.randint(-180, 180)
        pic1 = [TF.rotate(pic1[i], angle) for i in range(len(pic1))]
        pic2 = TF.rotate(pic2, angle)
    return pic1, pic2
def flip_lr_hr(pic1, pic2):
    if random.random() < 0.5:
        pic1 = [TF.hflip(pic1[i]) for i in range(len(pic1))]
        pic2 = TF.hflip(pic2)
    if random.random() < 0.5:
        pic1 = [TF.vflip(pic1[i]) for i in range(len(pic1))]
        pic2 = TF.vflip(pic2)
    return pic1, pic2

def random_crop(pic1, pic2, min_scale = 0.5, max_scale = 0.99):
    h_scale = np.random.uniform(low=min_scale, high=max_scale)
    w_scale = np.random.uniform(low=min_scale, high=max_scale)
    H, W = pic1[0].size[:2]
    y0 =   np.random.randint(0, H-H*h_scale)
    y1 =   int(H*h_scale)
    x0 =   np.random.randint(0, W-W*w_scale)
    x1 =   int(H*w_scale)
    pic1 = [TF.crop(pic1[i], y0, x0,y1, x1 ) for i in range(len(pic1))]
    pic2 = TF.crop(pic2, y0*3, x0*3,y1*3, x1*3)    

    return pic1, pic2
def do_gamma(pic1, pic2, gamma=1.0):
    pic1 = [np.clip(pic1[i] ** (1.0 / gamma), 0, 2) for i in range(len(pic1))]
    pic2 = np.clip(pic2 ** (1.0 / gamma), 0, 2)
    return pic1,pic2
def do_brightness_multiply(pic1, pic2, alpha=1):
    pic1 = [np.clip((alpha*pic1[i]), 0, 1) for i in range(len(pic1))]
    pic2 = np.clip((alpha*pic2), 0, 1)
    return pic1,pic2    