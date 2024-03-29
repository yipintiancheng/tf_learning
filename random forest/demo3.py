#随机森林2
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor 
import numpy as np 
from sklearn.datasets import load_iris 
iris=load_iris() 
 
from sklearn.model_selection import cross_val_score, ShuffleSplit 
X = iris["data"] 
Y = iris["target"] 
names = iris["feature_names"] 
rf = RandomForestRegressor(n_estimators=10) 
scores = [] 
for i in range(X.shape[1]): 
 score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2", 
 cv=ShuffleSplit(len(X), 3, .3)) 
 scores.append((round(np.mean(score), 3), names[i])) 
print(sorted(scores, reverse=True))
