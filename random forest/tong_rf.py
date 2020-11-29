# -*- coding: cp936 -*-
# 引入需要的包包
# 数据处理的常用包包
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

train_df=pd.read_csv ('F:/pycodes/o8.csv')
# 将DataFrame的数据转换成Array
train_data=train_df.values
np.random.shuffle(train_data)
array1=[]
clf=RandomForestClassifier(n_estimators=150, criterion='entropy', min_samples_split=2, min_weight_fraction_leaf=0.0,  max_features='sqrt')
cv = ShuffleSplit(n_splits=3, test_size=.3, random_state=0)
score=cross_val_score(clf, train_data[:,1:], train_data[:,0].ravel(), cv=cv)
print(np.mean(score))
array1=np.append(array1,np.mean(score))
del clf
del cv
del score
