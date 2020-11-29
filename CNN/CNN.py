#-*- coding:utf-8 -*-
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.utils import plot_model
import matplotlib.pyplot as plt
#from keras.utils import multi_gpu_model
import os
from read_data import get_data
#from get_tfr_data import get_data

from sklearn.cross_validation import cross_val_score # K折交叉验证模块

'消除警告'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

size = 64     #图片尺寸

def Model():
    # 搭建卷积神经网络
    model = Sequential()
    model.add(Conv2D(filters = 20,
                     kernel_size = (3, 3),
                     strides = (1,1),
                     padding = 'same',
                     input_shape = (size,size,1),
                     activation = 'relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # 重复构造，搭建深度网络
    model.add(Conv2D(50, kernel_size = (3, 3), strides = (1,1), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.5))

    # 把当前节点展平
    model.add(Flatten())

    # 构造全连接神经网络
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(0.5))

    #model.add(Dense(128, activation = 'relu'))
    model.add(Dense(600, activation = 'softmax'))  
    return model

def train_model(data,epochs,batch_size):

    x_train, y_train, x_test, y_test = data

    model = Model()
    '''
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    result = parallel_model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=epochs, batch_size=batch_size)
    scores = parallel_model.evaluate(x_test,y_test,verbose=0)
    '''
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    #result = model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=epochs, batch_size=batch_size)
    
    result = model.fit(x_train, y_train,validation_split=0.2, epochs=epochs, batch_size=batch_size)

    scores = model.evaluate(x_test,y_test,verbose=0) #评估模型成绩也就是验证及结果  
    print(scores)
    model.save('./../result_curve/02cnn_model.h5')
    plot_model(model, to_file='./../result_curve/02cnn_model.png')
    return result

def result_curve(result):
    # 绘制出结果
    plt.figure
    plt.subplot(121)
    plt.plot(result.epoch,result.history['acc'],label="acc")
    plt.plot(result.epoch,result.history['val_acc'],label="val_acc")
    plt.scatter(result.epoch,result.history['acc'])
    plt.scatter(result.epoch,result.history['val_acc'])
    plt.legend(loc='lower right')
    plt.title("CNN")

    plt.subplot(122)
    plt.plot(result.epoch,result.history['loss'],label="loss")
    plt.plot(result.epoch,result.history['val_loss'],label="val_loss")
    plt.scatter(result.epoch,result.history['loss'],marker='*')
    plt.scatter(result.epoch,result.history['val_loss'],marker='*')
    plt.legend(loc='upper right')
    plt.title("CNN")
    plt.savefig('./../result_curve/02cnn_curve.svg')
    plt.show()

def main(epochs,batch_size):
    data = get_data()
    result = train_model(data, epochs, batch_size)
    result_curve(result)

if __name__ =="__main__":
    main( epochs=130, batch_size=64)
