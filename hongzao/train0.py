#cd F:\drawprodoc\python\MeLearn;conda activate tf2.0;tensorboard --logdir logs
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import csv
#绘制示例
def plotImages(y,x):
	fig, axes = plt.subplots(1, 5, figsize=(10,10))
	for i in range(5):
		axes[i].set_title(x[i])
		axes[i].imshow(y[i,:,:,:])
		axes[i].axis('off')
	plt.tight_layout()
	plt.show()


IMG_HEIGHT = 224
IMG_WIDTH = 224
epochs = 20
fine_epochs = 20
batch_size = 32
#可调参数，记得更改保存路径
learning_rate = 0.0015#+0.0005
decay = 2
decay2 = 0.5
regularate = 0.0005#-0.00005
#全连接层结构，xception一共133个层
fine_tune_at = 80
dp = 0.5
n_dense = 96
dpp = 0.5
n_dense2 = 8
dppp = 0.4

import pathlib
data_root = pathlib.Path('F:/Image/train')
all_image_paths = list(data_root.glob('*/*'))
total_train = len(all_image_paths)
data_roott = pathlib.Path('F:/Image/test')
all_image_pathst = list(data_roott.glob('*/*'))
total_val = len(all_image_pathst)
#待定内容（拆分数据集）
# label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
# label_to_index = dict((name, index) for index, name in enumerate(label_names))
#for item in data_root.iterdir():
#  print(item)
#print(label_names) ['gantiao', 'huangpi', 'meibain', 'niaozhuo', 'wuquexian']
#print(label_to_index) {'gantiao': 0, 'huangpi': 1, 'meibain': 2, 'niaozhuo': 3, 'wuquexian': 4}

train_dir = os.path.join('F:/Image/train')
validation_dir = os.path.join('F:/Image/test')
train_image_generator = ImageDataGenerator( rescale=1./255,horizontal_flip=True,vertical_flip=True,
											width_shift_range=.15, height_shift_range=.15,rotation_range=45,zoom_range=0.5)
validation_image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
														   directory=train_dir,
														   shuffle=True,
														   target_size=(IMG_HEIGHT, IMG_WIDTH),
														   class_mode='sparse')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
															  directory=validation_dir,
															  target_size=(IMG_HEIGHT, IMG_WIDTH),
															  class_mode='sparse')

# #检查数据集格式，并绘制5个图片看看
# sample_training_images, sample_training_labels = next(train_data_gen)#取出一个batch的ndarray格式数据
# ds=tf.data.Dataset.from_tensor_slices((sample_training_images, sample_training_labels))#将此batch转为dataset格式
# # for y in sample_training_images:
# #     print(y)#查看张量形状
# # for y,x in ds:#查看dataset标签形状
# #     print(x.numpy())
# plotImages(sample_training_images, sample_training_labels)

IMG_SHAPE = (IMG_HEIGHT,IMG_WIDTH,3)#用迅雷下载
#模型下载与训练
base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE,include_top=False,
											   pooling='avg',weights='imagenet')
base_model.trainable = False
#检查模型结构与输入输出
# print(base_model.summary())
# feature_batch = base_model(sample_training_images)
# print(feature_batch.shape)#(4, 2048)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(learning_rate,decay_steps=total_train// batch_size*5,
															 decay_rate=decay,staircase=False)#递减的学习率

#模型方案一
hiden_layer = Dense(n_dense,activation=None,kernel_regularizer=regularizers.l2(regularate))
hiden_layer2 = Dense(n_dense2,activation=None,kernel_regularizer=regularizers.l2(regularate/2))#/2
prediction_layer = Dense(5, activation=None,kernel_regularizer=regularizers.l2(regularate/5))#
model = tf.keras.Sequential([base_model,Dropout(dp),
							 hiden_layer,BatchNormalization(),Activation('relu'),Dropout(dpp),#
							 hiden_layer2,BatchNormalization(),Activation('relu'),Dropout(dppp),#
							 prediction_layer,BatchNormalization(),Activation('softmax')])#
# #模型方案二
# x = base_model.output
# # x = Dropout(dp)(x)
# x = Dense(n_dense,kernel_regularizer=regularizers.l2(regularate))(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Dropout(dpp)(x)
# x = Dense(n_dense2,kernel_regularizer=regularizers.l2(regularate))(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Dropout(dppp)(x)
# predictions = Dense(5, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),#lr=learning_rate
			  loss=tf.keras.losses.sparse_categorical_crossentropy,
			  metrics=['accuracy'])
print(model.summary())

time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# log_dir = os.path.join('logs',time)#是log路径的问题，是TF2在Win下的一个bug，参考stackoverflow解决办法
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=False, write_grads=False, write_images=True)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')#,baseline=None, restore_best_weights=False

#模型训练
history = model.fit(train_data_gen,#方案二model.fit_generator将弃用
					steps_per_epoch=total_train// batch_size,
					epochs=epochs,validation_data=val_data_gen,
					validation_steps=total_val// batch_size,#
					callbacks=[early_stopping],verbose=1)#tensorboard_callback
# tf.saved_model.save(model,"saved/2")#模型保存

#可视化模型
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot( acc, label='Training Accuracy')
# plt.plot( val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# plt.subplot(1, 2, 2)
# plt.plot( loss, label='Training Loss')
# plt.plot( val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

#模型的微调
print('开始微调')
base_model.trainable = True
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False
print("Number of finetuned layers in the base model: ", len(base_model.layers)-fine_tune_at)
fine_lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(learning_rate/10,decay_steps=total_train// batch_size*10,
															 decay_rate=decay2,staircase=False)#递减的学习率
model.compile(optimizer=tf.keras.optimizers.Adam(fine_lr_schedule),#lr=learning_rate/10#模型须重新编译才能使layers的解冻有效
			  loss=tf.keras.losses.sparse_categorical_crossentropy,
			  metrics=['accuracy'])
#print(model.summary())

#timee=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#log_dirr = os.path.join('logs',timee)#是log路径的问题，是TF2在Win下的一个bug，参考stackoverflow解决办法
#tensorboard_callbackk = tf.keras.callbacks.TensorBoard(log_dir=log_dirr, histogram_freq=1,write_graph=True, write_grads=False, write_images=True)
pre_train = history.epoch[-1]
early_stoppingg = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.004, patience=1, verbose=0, mode='max')#,baseline=None, restore_best_weights=False
history_fine = model.fit(train_data_gen,steps_per_epoch=total_train// batch_size,
						 epochs=pre_train+fine_epochs+1,initial_epoch = pre_train+1,
						 validation_data=val_data_gen,validation_steps=total_val // batch_size,#
						 callbacks=[early_stoppingg],verbose=1)# callbacks=[tensorboard_callbackk],
tf.saved_model.save(model,"saved/39")#模型保存#
#model.save('hongzao_2.h5')

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

#训练结果写入F盘下的csv文件
accr=['train_accuracy:']+acc
val_accr=['valid_accuracy:']+val_acc
lossr=['train_loss:']+loss
val_lossr=['valid_loss:']+val_loss
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
with open('F:/train_log.csv','a',newline='') as f:
	f_csv = csv.writer(f)
	f_csv.writerow(['记录时间：',time,'n_dense=',n_dense,n_dense2,'learate=',learning_rate,'decay=',decay,decay2,'dropout=',dp,dpp,dppp,'regular=',regularate,'finetune=',133-fine_tune_at])
	f_csv.writerows([lossr])
	f_csv.writerows([accr])
	f_csv.writerows([val_lossr])
	f_csv.writerows([val_accr])

#训练结果可视化
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.1,1])
plt.plot([pre_train,pre_train],
		  plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0,2.0])
plt.plot([pre_train,pre_train],
		 plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
