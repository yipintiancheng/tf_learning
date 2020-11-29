import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib

#载入测试集路径，分离出图片路径列表与标签列表
def load(p):
    data_root = pathlib.Path(p)
    paths = list(data_root.glob('*/*'))
    paths = [str(path) for path in paths]
    labels = [label_to_index[pathlib.Path(path).parent.name] for path in paths]
    return paths,labels
#随机取n个结果
def random_take(labels,paths,n):
    indexs = np.random.randint(0, len(paths), size=n)  # 生成随机数
    blabels = np.array(labels)  # 标签列表转为数组
    bpaths = np.array(paths)  # 路径列表转为数组
    clabels = blabels[indexs]  # 取出数组中索引对应的部分标签
    cpaths = bpaths[indexs]  # 取出数组中索引对应的部分路径
    clabels.tolist()  # 部分标签转为列表
    cpaths.tolist()  # 部分路径转为列表
    return clabels,cpaths
#读取test文件夹图片
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0  # normalize to [0,1] range
    return image
#绘制被预测图像
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
#绘制预测概率图
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(5), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
#少量展示x个测试结果
def show(all_image_paths,all_image_labels,x):
    all_image_labels,all_image_paths=random_take(all_image_labels,all_image_paths,x)
    imds=[load_and_preprocess_image(path)for path in all_image_paths]#路径转为图片数组
    predictions=model(imds)#预测数据
    num_rows = 5
    num_cols = 4
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(x):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], all_image_labels, imds)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], all_image_labels)
    plt.tight_layout()
    plt.show()
#计算准确率
def accccuracy(all_image_paths,all_image_labels):
    batch_size = 32
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(len(all_image_paths) // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        imdss = [load_and_preprocess_image(path) for path in all_image_paths[start_index: end_index]]
        y_pred = model(imdss)
        sparse_categorical_accuracy.update_state(y_true=all_image_labels[start_index: end_index], y_pred=y_pred)
    b=sparse_categorical_accuracy.result()
    print("test accuracy: %f" % b)
    return b
#坏枣被当成好枣的预测结果展示
def resultshow(all_paths,all_labels):
    print('坏枣被当成好枣个数：',len(l),'这种坏枣比例：',len(l)/len(all_labels), '\n序号列表：', l,'\n绘制其中20个图像')
    m = l[0:20]
    all_image_labels=[]
    all_image_paths=[]
    for i in m:
        all_image_labels.append(all_labels[i])
        all_image_paths.append(all_paths[i])
    imds = [load_and_preprocess_image(path) for path in all_image_paths]  # 路径转为图片数组
    predictions = model(imds)  # 预测数据
    num_rows = 5
    num_cols = 4
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(20):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], all_image_labels, imds)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], all_image_labels)
    plt.tight_layout()
    plt.show()
def bad_accccuracy(all_image_paths,all_image_labels):
    batch_size = 32
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(len(all_image_paths) // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        imdss = [load_and_preprocess_image(path) for path in all_image_paths[start_index: end_index]]
        y_pred = model(imdss)
        for p in range(tf.shape(y_pred)[0]):
            if np.argmax(y_pred[p]) == 4:
                l.append(start_index+p)
        sparse_categorical_accuracy.update_state(y_true=all_image_labels[start_index: end_index], y_pred=y_pred)
    b=sparse_categorical_accuracy.result()
    print("test accuracy: %f" %b)
    return b

label_to_index={'gantiao': 0, 'huangpi': 1, 'meibian': 2, 'potou': 3, 'zhengchang': 4}
class_names=   {0:'gantiao', 1:' huangpi',  2:'meibian', 3 :'potou',  4:'zhengchang'}
model=tf.saved_model.load("saved/48")#导入图文件(文件比较小)
#model = tf.keras.models.load_model('hongzao_1.h5')#或者导入二进制blob文件HDF5
path='F:/Imagee/test'#Image全部枣，Imagee好枣，Imageg坏枣
l = []#储存被当成好枣的坏枣序号

if __name__=="__main__":
    paths, labels = load(path)#生成标签
    possibility = bad_accccuracy(paths, labels)#
    show(paths,labels,20)#随机绘制20个结果，或者少于20个也行
    # resultshow(paths,labels)#搭配bad_accccuracy使用

