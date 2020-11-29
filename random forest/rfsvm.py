#!/usr/bin/env python
# coding=utf8
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from IPython.display import display
from sklearn import metrics
from sklearn import cross_validation


if __name__ == "__main__":
    train_df=pd.read_csv ('train.csv')
    train_data=train_df.values
    np.random.shuffle(train_data)
    np.random.shuffle(train_data)
    np.random.shuffle(train_data)
    num_features=train_data.shape[0]
    split=int(num_features*2/3)
    train = train_data[:split]
    test = train_data[split:]

    pipe_svc = Pipeline([('scl', StandardScaler()),
                         ('clf', SVC())])
    C_range = np.logspace(-3, 3, 7)
    gamma_range = np.logspace(-3, 3, 7)
    param_grid = [
        {'clf__C': C_range, 'clf__gamma': gamma_range, 'clf__kernel': ['rbf']}  # 对于核SVM则需要同时调优C和gamma值
    ]
    


    parameters=[{'n_estimators':[100,200,300,400,500,600],'criterion':['entropy']}]
    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      cv=3,
                      n_jobs=-1)

    clf=GridSearchCV(RandomForestClassifier(), parameters,cv=3,n_jobs=-1)
    clf.fit(train[:,1:], train[:,0].ravel())
    gs.fit(train[:,1:], train[:,0].ravel())
    print(clf.best_score_)
    print(clf.best_params_)
    print(gs.best_score_)
    print(gs.best_params_)
    output=clf.predict(test[:,1:])
    outputsvm=gs.predict(test[:,1:])
    print (metrics.confusion_matrix(test[:,0].ravel(), output))
    print (metrics.accuracy_score(test[:,0].ravel(), output))

    print("SVM:")

    print (metrics.confusion_matrix(test[:,0].ravel(), outputsvm))
    print (metrics.accuracy_score(test[:,0].ravel(), outputsvm))


