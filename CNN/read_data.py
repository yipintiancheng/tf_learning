#!/usr/bin/env python
# coding=utf-8
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import random

def read_img(img_name):
    im = Image.open(img_name)
    # im = im.resize((32,32),Image.ANTIALIAS)
    # im = im.convert('RGB')
    data = np.array(im)
    return data

def read_label(img_name):
    basename = os.path.basename(img_name)
    label = basename.split('_')[0]
    return label

def get_data(shuffle=True):
    pt = "./../data"
    images, labels = [], []
    for x in os.listdir(pt):
        d = os.path.join(pt,x)
        images.append(read_img(d))
        labels.append(read_label(d))

    if shuffle==True:
        imgs, lbs = [], []
        index = [x for x in range(len(labels))]
        random.shuffle(index)
        for i in index:
            imgs.append(images[i])
            lbs.append(labels[i])
        images, labels = imgs, lbs

    y = np.array(list(map(int,labels)))
    x = np.array(images)
    x = x.reshape(len(x),64,64,1)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

    y_train = np_utils.to_categorical(y_train,600)
    y_test = np_utils.to_categorical(y_test,600)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # im = x_test[285].reshape(32,32)*255
    # img = Image.fromarray(im)
    # img.show()
    # print(im,y_test[285])

    return (x_train,y_train,x_test,y_test)
