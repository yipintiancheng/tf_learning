import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pathlib
import datetime
import csv
import os
drop=0.4
lr=0.01

# #绘制图象pyinstaller -F -w -i m.ico GUI.py
# acc = [0.3596059,0.48029557,0.54351395,0.5837439,0.58538586,0.6096059,0.6141215,0.62356323,0.6699507,0.7192118,0.7458949,0.76724136,0.7680624,0.77483976,0.7844828,0.7803777,0.80090314,0.80448717]
# val_acc = [0.3125,0.29464287,0.2767857,0.3169643,0.328125,0.3482143,0.33035713,0.3549107,0.45535713,0.56026787,0.61383927,0.67410713,0.75,0.7566964,0.7433036,0.7745536,0.77008927,0.7410714]
# loss = [0.3125,0.29464287,0.2767857,0.3169643,0.328125,0.3482143,0.33035713,0.3549107,0.45535713,0.56026787,0.61383927,0.67410713,0.75,0.7566964,0.7433036,0.7745536,0.77008927,0.7410714]
# val_loss = [1.6057354722704207,1.5644276482718331,1.5737601518630981,1.533537404877799,1.5601051194327218,1.520599637712751,1.5782304491315569,1.5219760622297014,1.3504310165132796,1.2154205696923392,1.188516684940883,1.1373544250215804,1.1250545978546143,1.140255229813712,1.107681461742946,1.0960354038647242,1.1020221199308122,1.136080418]
# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.ylim([0.1,1])
# plt.plot([8-1,8-1],
#           plt.ylim(), label='Start Fine Tuning')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.ylim([0,2.0])
# plt.plot([8-1,8-1],
#          plt.ylim(), label='Start Fine Tuning')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()


#写入文件
#acc.insert(0,'train_accuracy')
#val_acc.insert(0,'valid_accuracy')
#loss.insert(0,'train_loss')
#val_loss.insert(0,'valid_loss')
#time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#with open('F:/train_log.csv','a',newline='') as f:
#    f_csv = csv.writer(f)
#    f_csv.writerow([time,'dorp',drop,'lr',lr])
#    f_csv.writerows([loss])
#    f_csv.writerows([acc])
#    f_csv.writerows([val_loss])
#    f_csv.writerows([val_acc])


#读取文件个数，创建标签字典#查看学习率衰减速率
data_root = pathlib.Path('F:/Image/train')
all_image_paths = list(data_root.glob('*/*'))
# label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
# label_to_index = dict((name, index) for index, name in enumerate(label_names))
# for item in data_root.iterdir():
#   print(item)
# print(label_names)# ['gantiao', 'huangpi', 'meibain', 'niaozhuo', 'wuquexian']
# print(label_to_index)# {'gantiao': 0, 'huangpi': 1, 'meibain': 2, 'niaozhuo': 3, 'wuquexian': 4}

total_train = len(all_image_paths)
batch_size = 32
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.0002,decay_steps=total_train// batch_size,
                                                             decay_rate=0.2,staircase=False)
s=total_train// batch_size
step = np.linspace(0,400)
lr = lr_schedule(step)
plt.figure(figsize = (8,6))
plt.plot(step/s, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')
plt.show()
# import datetime
# starttime = datetime.datetime.now()
# #long running
# endtime = datetime.datetime.now()
# print((endtime - starttime).seconds)
# print('yunxing'+str((endtime - starttime).seconds)+'s')