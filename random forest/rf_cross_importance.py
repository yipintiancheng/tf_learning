# -*- coding: cp936 -*-
# ������Ҫ�İ���
# ���ݴ���ĳ��ð���
import numpy as np
import pandas as pd

# ���ɭ�ֵİ���
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
# ��ͼ�İ���
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

# ��ȡ��DataFrame������
train_df=pd.read_csv ('orderfeatureimportance.csv')
# ��DataFrame������ת����Array
train_data=train_df.values
train_data=train_data[:,:29]

'''
# 2/3��train_data��Ϊѵ�����ݣ�1/3��train_data��Ϊ����������ѵ��ģ��
num_features=train_data.shape[0]# �õ�train_data��������Ҳ�����������ݵĸ�����Ϊ����ֵ
print ("Number of all features: \t\t",num_features)

n_samples=train_data.shape[0]
'''
# ��ʼʹ�����ɭ�ַ�����
clf=RandomForestClassifier(n_estimators=200)# ����������ĸ���Ϊ100

cv = ShuffleSplit(n_splits=3, test_size=.3, random_state=0)
score=cross_val_score(clf, train_data[:,1:], train_data[:,0].ravel(), cv=cv)
print (np.mean(score))

