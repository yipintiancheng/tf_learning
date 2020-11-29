# -*- coding: cp936 -*-
# 引入需要的包包
# 数据处理的常用包包
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

train_df=pd.read_csv ('orderfeatureimportance.csv')
# 将DataFrame的数据转换成Array
train_data=train_df.values
np.random.shuffle(train_data)
array1=[];

for i in range(21,128):
    traindata=train_data[:,:i]
    clf=RandomForestClassifier(n_estimators=200)
    cv = ShuffleSplit(n_splits=3, test_size=.3, random_state=0)
    score=cross_val_score(clf, train_data[:,1:], train_data[:,0].ravel(), cv=cv)
    print(i)
    print(np.mean(score))
    array1=np.append(array1,np.mean(score))
    del traindata
    del clf
    del cv
    del score
    

np.savetxt('E:\\1\\codes\\re.csv',array1,delimiter=',')
