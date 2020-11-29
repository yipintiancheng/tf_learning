#cd F:\drawprodoc\python\MeLearn;conda activate tf2.0;tensorboard --logdir logs
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

train = 'F:/Image/train'
valid = 'F:/Image/test'
IMG_HEIGHT = 224
IMG_WIDTH = 224
epochs = 10
fine_epochs = 20
batch_size = 32
starttime = datetime.datetime.now()
#可调参数，记得更改保存路径
mark=str(59)#降低正则率，降低学习率，降低正则比，
learning_rate = 0.0004
decay = 0.2
decay2 = 0.1
regularate = 0.00004
#全连接层结构，xception一共133个层
fine_tune_at = 80
dp = 0.4
n_dense = 96
dpp = 0.4
n_dense2 = 12
dppp = 0.4
#打印网络结构
print([mark,'n_dense=',n_dense,n_dense2,'learate=',learning_rate,'decay=',decay,decay2,'dropout=',dp,dpp,dppp,'regular=',regularate,'finetune=',133-fine_tune_at])

#图片个数
import pathlib
data_root = pathlib.Path(train)
all_image_paths = list(data_root.glob('*/*'))
total_train = len(all_image_paths)
data_roott = pathlib.Path(valid)
all_image_pathst = list(data_roott.glob('*/*'))
total_val = len(all_image_pathst)
#实例化生成器
train_dir = os.path.join(train)
validation_dir = os.path.join(valid)
train_image_generator = ImageDataGenerator( rescale=1./255,horizontal_flip=True,vertical_flip=True,width_shift_range=.15, height_shift_range=.15,rotation_range=45,zoom_range=0.5)
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
# plotImages(sample_training_images, sample_training_labels)

#模型下载与训练,用迅雷下载
IMG_SHAPE = (IMG_HEIGHT,IMG_WIDTH,3)
base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE,include_top=False,pooling='avg',weights='imagenet')
base_model.trainable = False
#构造模型
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(learning_rate,decay_steps=total_train// batch_size,decay_rate=decay,staircase=False)#递减的学习率
hiden_layer = Dense(n_dense,activation=None,kernel_regularizer=regularizers.l2(regularate/5))
hiden_layer2 = Dense(n_dense2,activation=None,kernel_regularizer=regularizers.l2(regularate/2))#/2
prediction_layer = Dense(5, activation=None,kernel_regularizer=regularizers.l2(regularate))#
model = tf.keras.Sequential([base_model,Dropout(dp),
							 hiden_layer,BatchNormalization(),Activation('relu'),Dropout(dpp),#
							 hiden_layer2,BatchNormalization(),Activation('relu'),Dropout(dppp),#
							 prediction_layer,BatchNormalization(),Activation('softmax')])#
#编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),#lr=learning_rate
			  loss=tf.keras.losses.sparse_categorical_crossentropy,
			  metrics=['accuracy'])
print(model.summary())
#模型训练
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='min',restore_best_weights=True)#,baseline=None
history = model.fit(train_data_gen,#odel.fit_generator
					steps_per_epoch=total_train// batch_size,
					epochs=epochs,validation_data=val_data_gen,
					validation_steps=total_val// batch_size,
					callbacks=[early_stopping],verbose=1)#tensorboard_callback
#记录训练过程
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

#模型的微调
print('开始微调')
base_model.trainable = True
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False
print("Number of finetuned layers in the base model: ", len(base_model.layers)-fine_tune_at)
#重新编译
fine_lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(learning_rate/10,decay_steps=total_train// batch_size,decay_rate=decay2,staircase=False)#递减的学习率
model.compile(optimizer=tf.keras.optimizers.Adam(fine_lr_schedule),#lr=learning_rate/10#模型须重新编译才能使layers的解冻有效
			  loss=tf.keras.losses.sparse_categorical_crossentropy,
			  metrics=['accuracy'])
pre_train = history.epoch[-1]
early_stoppingg = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.005, patience=5, verbose=1, mode='max',restore_best_weights=True)
history_fine = model.fit(train_data_gen,steps_per_epoch=total_train// batch_size,
						 epochs=pre_train+fine_epochs+1,initial_epoch = pre_train+1,
						 validation_data=val_data_gen,validation_steps=total_val // batch_size,
						 callbacks=[early_stoppingg],verbose=1)# callbacks=[tensorboard_callbackk],
#记录训练过程
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

#模型保存
pp="saved/"+mark
tf.saved_model.save(model,pp)

#训练结果写入F盘下的csv文件
accr=['train_accuracy:']+acc
val_accr=['valid_accuracy:']+val_acc
lossr=['train_loss:']+loss
val_lossr=['valid_loss:']+val_loss

endtime = datetime.datetime.now()
time = endtime.strftime("%Y%m%d-%H%M%S")
ts=str((endtime - starttime).seconds)

with open('F:/train_log.csv','a',newline='') as f:
	f_csv = csv.writer(f)
	f_csv.writerow([mark+'记录时间：',time,'运行时长:',ts+'s','n_dense=',n_dense,n_dense2,'learate=',learning_rate,'decay=',decay,decay2,'dropout=',dp,dpp,dppp,'regular=',regularate,'finetune=',133-fine_tune_at])
	f_csv.writerows([lossr])
	f_csv.writerows([accr])
	f_csv.writerows([val_lossr])
	f_csv.writerows([val_accr])
print('运行时长:'+ts+'s')

#训练结果可视化
plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.1,1])
plt.plot([pre_train,pre_train],plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0,2.0])
plt.plot([pre_train,pre_train],plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

plt.show()
