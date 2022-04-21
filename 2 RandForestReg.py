import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('modified5.csv')
dataset.head()
dataset.describe()

X = dataset.iloc[0:,3:]
Y = dataset['FLAG']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
clsfy = RandomForestClassifier(n_estimators=300, random_state=0) 
cls = clsfy.fit(X_train, Y_train)

Y_pred = clsfy.predict(X_test)

from sklearn import metrics
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(Y_test, Y_pred))

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(Y_test, Y_pred))
confusion = confusion_matrix(Y_test, Y_pred)
print(confusion)
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]
print('Sensitivity:', TP / float(TP+FN))
print('Specificity:', TN / float(TN+FP))

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))











