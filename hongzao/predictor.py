import tensorflow as tf
import numpy as np

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0  # normalize to [0,1] range
    return image
def pred(path):
    imarry=[load_and_preprocess_image(path)]#路径转为图片数组[1,244,244,3]
    predictions=model(imarry)#预测数据
    predic = class_names[np.argmax(predictions)]  # 转为类别名称
    pp = np.max(predictions)
    print("预测结果是%s,概率是%f"%(predic,pp))
    return predic,pp
def mainloop():
    while True:
        path = input('请输入图片路径(回车以结束):')
        if path == '':
            print('预测程序结束')
            break
        else:
            pred(path)

class_names = {0:'gantiao', 1:'huangpi',  2:'meibian', 3 :'potou',  4:'zhengchang'}
model=tf.saved_model.load("saved/29")
#model = tf.keras.models.load_model('hongzao_1.h5')

if __name__=='__main__':
    mainloop()#F:\Image\test\gantiao\gt.500.jpg
