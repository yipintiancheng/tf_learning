#-*- coding:utf-8 -*-
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model
import os
from read_data import get_data
#from get_tfr_data import get_data

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adadelta, Adagrad

from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.models import load_model
import h5py

import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import random

from keras.preprocessing import image



def read_img(img_name):
    im = Image.open(img_name)
    # im = im.resize((32,32),Image.ANTIALIAS)
    # im = im.convert('RGB')
    data = np.array(im)
    return data


def test():
    '消除警告'
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    
    data = get_data() 
    x_train, y_train, x_test, y_test = data
    images = []
    images.append(read_img(r'./../data/00004_09.bmp'))
    x = np.array(images)
    x = x.reshape(len(x),64,64,1)
    x = x.astype('float32')
    x /= 255

    model = load_model('./../result_curve/AlexNet_model.h5')

    scores = model.evaluate(x_test,y_test,verbose=0) #评估模型成绩也就是验证及结果  

    preds = model.predict(x)  #评估图片x属于各个类的概率  
    preds_class = model.predict_classes(x)  #评估图片属于哪一类 
    print('score is: ',scores)
    print('preds: ',preds)
    print('preds_class:', preds_class)
    
    return preds_class

test()




