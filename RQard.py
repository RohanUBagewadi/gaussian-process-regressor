import pyGPs as Gps
import numpy as np
import pandas as pd

# Importing the datasets

datasets = pd.read_csv('data/Sogevac_65_hot.csv')
X = datasets.iloc[:, :3].values
Y = datasets.iloc[:, 3].values

x_new = np.array(X[50:60,:])
x_new = np.reshape(x_new, (-1, 3))

datasets_2 = pd.read_csv('data/')

model = Gps.GPR()

k = Gps.cov.RQard(D=X.shape[1])
m = Gps.mean.Linear(D=X.shape[1])

model.setData(x=X, y=Y)
model.setPrior(mean= m, kernel=k)
model.optimize(X, Y)
print('Model fitted')

y_new = model.predict(X)
y_pred = np.array(y_new[0])
print(max(y_pred-np.reshape(Y, (-1, 1))))