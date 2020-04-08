import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import forest
import numpy as np
import json

data1 = pd.read_csv('sdss.csv')
data = pd.read_csv('sdss_redshift.csv')
X = data[['u', 'g', 'r', 'i', 'z']]
y = data['redshift']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
params = {'n_estimators': np.arange(1,11), 'max_depth': np.arange(1,11)}
clf =  RandomForestRegressor()
clf_grid = GridSearchCV(clf, params, cv=5, n_jobs=-1)
clf_grid.fit(X_train, y_train)
p = clf_grid.best_params_


ctg = forest.DecisionForest(X_train.values, y_train.values, p['max_depth'], p['n_estimators'])
ctg.fit1(X_train.values, y_train.values)
y_pred = ctg.predict1(X.values)
y_test_pred = ctg.predict1(X_test.values)
y_train_pred = ctg.predict1(X_train.values)
data = {
  "train": np.sqrt(mean_squared_error(y_train, y_train_pred)),
  "test": np.sqrt(mean_squared_error(y_test, y_test_pred))
  }
with open("redshift.json", "w") as f:
    json.dump(data, f)

y_pred1 = pd.DataFrame(ctg.predict1(data1.values))
data1['redshift'] = y_pred1
data1.to_csv(r'sdss_predict.csv', index=None)
plt.scatter(y, y_pred)
plt.title('истинное значение — предсказание')
plt.xlabel('$y_{true}$')
plt.ylabel('$y_{pred}$')
plt.savefig('redshift.png')
