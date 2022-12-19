# POLYNOMIAL LINEAR REGRESSION

# Required Libraries
import pandas as pd
import numpy as np

# Loading datset
data = pd.read_csv("Data.csv")
X = data.iloc[:,2:-1].values
Y = data.iloc[:,-1].values

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

# Training the model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree=4)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.fit_transform(X_test)
model = LinearRegression()
model.fit(X_poly_train, Y_train)

# Testing the model
Y_pred = model.predict(X_poly_test)
np.set_printoptions(precision=2)
#print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))

# Evaluating the model 
from sklearn.metrics import r2_score
print(r2_score(Y_test, Y_pred))