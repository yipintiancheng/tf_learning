import sys
from PyQt5 import QtCore,QtWidgets,QtGui
import tensorflow as tf
import numpy as np
import cv2

class hongzao(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头
        self.class_names = {0: 'gantiao', 1: 'huangpi', 2: 'meibian', 3: 'potou', 4: 'zhengchang'}
        self.set_ui()
        self.show()

    def set_ui(self):
        self.resize(840, 400)
        self.setWindowTitle("预测红枣类别")
        self.label = QtWidgets.QLabel(self)#定义标签1
        self.label.setText("  显示图片区域")
        self.label.setFixedSize(280, 280)
        self.label.move(180, 105)
        self.label.setStyleSheet("QLabel{background:white;}""QLabel{color:rgb(300,300,300,120);font-size:15px;font-weight:bold;font-family:宋体;}")
        self.label2 = QtWidgets.QLabel(self)#定义标签2
        self.label2.setText("  显示结果区域")
        self.label2.setFixedSize(280, 50)
        self.label2.move(180, 30)
        self.label2.setStyleSheet("QLabel{background:pink;}")
        self.label_show_camera = QtWidgets.QLabel(self)
        self.label_show_camera.setFixedSize(321, 241)
        self.label_show_camera.setStyleSheet("QLabel{background:white;}")
        self.label_show_camera.move(490, 80)
        self.btn = QtWidgets.QPushButton(self)#定义按钮1
        self.btn.setText("打开图片")
        self.btn.move(40, 140)
        self.btn.clicked.connect(self.openimage)  # 连接信号和槽函数
        self.button_open_camera = QtWidgets.QPushButton(self)
        self.button_open_camera.setText("打开相机")
        self.button_open_camera.move(40, 180)
        self.button_open_camera.clicked.connect(self.button_open_camera_clicked)
        self.button_camera = QtWidgets.QPushButton(self)
        self.button_camera.setText("拍摄图片")
        self.button_camera.move(40, 220)
        self.button_camera.clicked.connect(self.predict_camera)
        self.timer_camera.timeout.connect(self.show_camera)

    def openimage(self):
        imgName, imgType = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        predic,pp = self.pred(imgName)
        self.label2.setText("预测结果:%s  概率:%f" % (predic, pp))

    def button_open_camera_clicked(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open_camera.setText('关闭相机')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.label_show_camera.clear()  # 清空视频显示区域
            self.button_open_camera.setText('打开相机')

    def show_camera(self):
        flag, self.image = self.cap.read()  # 从视频流中读取
        show = cv2.resize(self.image, (320, 320))  # 把读到的帧的大小重新设置
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        self.im = show
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage

    def predict_camera(self):
        show = self.im
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        showImage = showImage.scaled(self.label.width(), self.label.height())
        self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
        mm = tf.image.resize(show, [224, 224])
        mm /= 255.0
        predictions = model([mm])  # 预测数据
        predic = self.class_names[np.argmax(predictions)]  # 转为类别名称
        pp = np.max(predictions)
        self.label2.setText("预测结果:%s  概率:%f" % (predic, pp))

    def pred(self,path):
        if path == '':
            predic = '无文件'
            pp = 0.0
        else:
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [224, 224])# normalize to [0,1] range  
            predictions = model([image/255.0])  # 预测数据
            predic = self.class_names[np.argmax(predictions)]  # 转为类别名称
            pp = np.max(predictions)
        return predic, pp

if __name__=='__main__':
    model = tf.saved_model.load("saved/48")
    app = QtWidgets.QApplication(sys.argv)
    m = hongzao()
    sys.exit(app.exec_())
