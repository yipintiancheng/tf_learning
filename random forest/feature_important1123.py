#!/usr/bin/env python
# coding=utf8

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)

train_df=pd.read_csv ('train.csv')
train_data=train_df.values
np.random.shuffle(train_data)#打乱，打乱，打乱，打乱，打乱，打乱，打乱，打乱，打乱


num_features=train_data.shape[0]
print ("Number of all features: \t\t",num_features)
train = train_data

X=train[:,1:]
Y=train[:,0].ravel()
print("Number of features used for training: \t", len(train))

for i in range(len(train)):
    sample=train[i]
    for j in range(len(sample)):
        if np.isnan(sample[j]):
            sample[j]=0

rf = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=100, min_samples_split=2, min_weight_fraction_leaf=0.0)
scores = defaultdict(list)
names=["band1","band11","band12","band2","band3","band4","band5","band6","band7","band8","band8a","band9","amplitude_s1","basevalue_s1","endseason_s1","largeintegral_s1","leftderivative_s1","lengthseason_s1","maxfitdata_s1","positionmseason_s1","rightderivative_s1","smallintegral_s1","startseason_s1","mean_b1","mean_b11","mean_b12","mean_b2","mean_b3","mean_b4","mean_b5","mean_b6","mean_b7","mean_b8","mean_b8a","mean_b9","variance_b1","variance_b11","variance_b12","variance_b2","variance_b3","variance_b4","variance_b5","variance_b6","variance_b7","variance_b8","variance_b8a","variance_b9","homogeneity_b1","homogeneity_b11","homogeneity_b12","homogeneity_b2","homogeneity_b3","homogeneity_b4","homogeneity_b5","homogeneity_b6","homogeneity_b7","homogeneity_b8","homogeneity_b8a","homogeneity_b9","contrast_b1","contrast_b11","contrast_b12","contrast_b2","contrast_b3","contrast_b4","contrast_b5","contrast_b6","contrast_b7","contrast_b8","contrast_b8a","contrast_b9","dissimilarity_b1","dissimilarity_b11","dissimilarity_b12","dissimilarity_b2","dissimilarity_b3","dissimilarity_b4","dissimilarity_b5","dissimilarity_b6","dissimilarity_b7","dissimilarity_b8","dissimilarity_b8a","dissimilarity_b9","entropy_b1","entropy_b11","entropy_b12","entropy_b2","entropy_b3","entropy_b4","entropy_b5","entropy_b6","entropy_b7","entropy_b8","entropy_b8a","entropy_b9","secondmoment_b1","secondmoment_b11","secondmoment_b12","secondmoment_b2","secondmoment_b3","secondmoment_b4","secondmoment_b5","secondmoment_b6","secondmoment_b7","secondmoment_b8","secondmoment_b8a","secondmoment_b9","correlation_b1","correlation_b11","correlation_b12","correlation_b2","correlation_b3","correlation_b4","correlation_b5","correlation_b6","correlation_b7","correlation_b8","correlation_b8a","correlation_b9","EVI","GCVI","LSWI","NDVI","NDVI705","mNDVI705","mSR705"]
for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    r = rf.fit(X_train, Y_train)
    acc = r2_score(Y_test, rf.predict(X_test))
    for i in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(Y_test, rf.predict(X_t))
        scores[names[i]].append((acc-shuff_acc)/acc)
print ("Features sorted by their score:")
print (sorted([(round(np.mean(score), 4), feat) for
              feat, score in scores.items()], reverse=True))
