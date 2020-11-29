import tensorflow as tf
import numpy as np

#预处理数据
class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)  # [60000]
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]

#定义模型类
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,  # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],  # 感受野大小
            padding='same',  # padding策略（vaild 或 same）
            activation=tf.nn.relu  # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)  # [batch_size, 28, 28, 32]
        x = self.pool1(x)  # [batch_size, 14, 14, 32]
        x = self.conv2(x)  # [batch_size, 14, 14, 64]
        x = self.pool2(x)  # [batch_size, 7, 7, 64]
        x = self.flatten(x)  # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)  # [batch_size, 1024]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output

batch_size = 50#一组的数据个数
model = CNN()
data_loader = MNISTLoader()
#以上复制原模型
checkpoint=tf.train.Checkpoint(mm=model)#实例化检查点
checkpoint.restore(tf.train.latest_checkpoint('./save'))#恢复参数

#测试训练结果
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()                                              #实例化评估器
num_batches = int(data_loader.num_test_data // batch_size)                                                              #设定测试组数
for batch_index in range(num_batches):                                                                                 #遍历测试集每一个数据
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size                                     #设定自动向后推移
    y_pred = model.predict(data_loader.test_data[start_index: end_index])                                                 #计算本组数据预测值
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)        #保存累计输入过的数据，用于最后计算总的准确率
print("test accuracy: %f" % sparse_categorical_accuracy.result())                                                     #sparse_categorical_accuracy.result()用于输出训练正确的结果的比例