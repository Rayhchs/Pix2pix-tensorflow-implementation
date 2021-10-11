"""
Created on Wed Sep 8 2021

utils

@author: Ray
"""
import os, random
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.preprocessing import image

def random_crop(images, labels):
    from tensorflow.keras.preprocessing import image
    inputs = np.concatenate((images, labels), axis=2)
    ranges = random.uniform(0.8, 1)
    outputs = image.random_zoom(inputs, zoom_range=(ranges,ranges), row_axis=0, col_axis=1, channel_axis=2)
    image = outputs[:,:,0:3]
    label = outputs[:,:,3:6]
    return image, label

def load_data(images, mode):
    g_imgs = []
    d_imgs = []
    
    if mode == 'train':
        for i in range(len(images)):
            g_train, d_train = images[i].split(', ')
            g_img = np.array(image.load_img(g_train, target_size=(256, 256)))
            d_img = np.array(image.load_img(d_train, target_size=(256, 256)))
            g_img, d_img = random_crop(g_img, d_img)
            g_img = (g_img / 127.5) - 1
            d_img = (d_img / 127.5) - 1
            g_imgs.append(g_img)
            d_imgs.append(d_img)
            
    elif mode == 'test':
        for i in images:
            g_img = np.array(image.load_img(i, target_size=(256, 256)))
            g_img = (g_img / 127.5) - 1
            g_imgs.append(g_img)
        
    return g_imgs, d_imgs
        
def resize(o_img_path, g_img):
    o_img = np.array(image.load_img(o_img_path))
    m, n, _ = o_img.shape
    g_img = tf.image.resize(g_img, [m, n])

    return g_img
    
def save_data(im, images, save_path):
    if save_path == None:
        save_path = os.getcwd() + '\\result'
        for i in range(len(images)):
            os.mkdir(os.getcwd() + '\\result') if os.path.exists(os.getcwd() + '\\result') == False else None
            im[i].save(save_path + '\\' + os.path.basename(images[i]))
                
    else:
        for i in range(len(images)):
            im[i].save(save_path + '\\' + os.path.basename(images[i])) if os.path.exists(save_path) == True else sys.exit('Dir not exist')

    return save_path