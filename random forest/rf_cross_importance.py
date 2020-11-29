# -*- coding: cp936 -*-
# 引入需要的包包
# 数据处理的常用包包
import numpy as np
import pandas as pd

# 随机森林的包包
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
# 画图的包包
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

# 读取成DataFrame的数据
train_df=pd.read_csv ('orderfeatureimportance.csv')
# 将DataFrame的数据转换成Array
train_data=train_df.values
train_data=train_data[:,:29]

'''
# 2/3的train_data作为训练数据，1/3的train_data作为测试数据来训练模型
num_features=train_data.shape[0]# 拿到train_data的行数，也就是所有数据的个数作为特征值
print ("Number of all features: \t\t",num_features)

n_samples=train_data.shape[0]
'''
# 开始使用随机森林分类器
clf=RandomForestClassifier(n_estimators=200)# 定义决策树的个数为100

cv = ShuffleSplit(n_splits=3, test_size=.3, random_state=0)
score=cross_val_score(clf, train_data[:,1:], train_data[:,0].ravel(), cv=cv)
print (np.mean(score))

