#  SUPPORT VECTOR REGRESSION

# Required Libraries
import pandas as pd
import numpy as np

# Loading datset
data = pd.read_csv("Data.csv")
X = data.iloc[:,2:-1].values
Y = data.iloc[:,-1].values
Y = Y.reshape(len(Y),1)
# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
Y_train = sc_Y.fit_transform(Y_train)
X_test = sc_X.fit_transform(X_test)

# Training the model
from sklearn.svm import SVR
model = SVR(kernel='rbf')
model.fit(X_train, Y_train)

# Testing the model
Y_pred = model.predict(X_test)
Y_pred = sc_Y.inverse_transform(Y_pred.reshape(-1,1))
np.set_printoptions(precision=2)
#print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))

# Evaluating the model 
from sklearn.metrics import r2_score
print(r2_score(Y_test, Y_pred))
