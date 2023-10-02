# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: JEEVITHA S
RegisterNumber:  212222100016
*/

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)
```

## Output:
### Array Value of x:
![233685454-0adb4b94-f3fd-4e97-b8aa-6fcefb043d74](https://github.com/Jeevithha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123623197/83ebb00b-1c57-402b-a2f7-0658a02fca23)
### Array Value of y:
![233685521-644c69ed-e33f-4c87-9328-fc0bef73714c](https://github.com/Jeevithha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123623197/b4923b23-5ea0-4438-9ff2-3d5b26ea0380)
### Exam 1 - score graph:
![233688106-0246eac6-0a24-4b94-b162-1c08c4777107](https://github.com/Jeevithha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123623197/fcd3106a-6d7c-486c-9c1f-254e7700906b)
### Sigmoid function graph:
![233685832-e45b2e1b-aaf0-4828-bb22-eb845e32030b](https://github.com/Jeevithha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123623197/33392058-1278-40b9-93d1-287451f73f82)
### X_train_grad value:
![233686027-0877fcff-43e1-458b-9c76-775fd2c09e98](https://github.com/Jeevithha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123623197/a2008bce-00a6-4006-b1d1-c28603675a86)
### Y_train_grad value:
![233686071-d4e01b9b-5bf1-4b47-b546-9f82e30c6213](https://github.com/Jeevithha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123623197/d0f55626-1929-4f14-acd0-f84507a6b903)
### Print res.x:
![233686143-e28eb2d2-b7b7-42ff-9ce9-6009bd8ebfc5](https://github.com/Jeevithha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123623197/8b7eb96e-a2c7-40a6-9da2-7bac08e0ee8b)
### Decision boundary - graph for exam score:
![233686223-36c51815-9370-4076-885a-85c553eee393](https://github.com/Jeevithha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123623197/7c705a67-462a-421a-b0f3-35d9c2d86c58)
### Proability value:
![233686314-7faa0663-8a83-49a7-9194-3941b2733b51](https://github.com/Jeevithha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123623197/2387b993-0a7b-4103-b8f0-e2db18757463)
### Prediction value of mean:
![233686372-15b81f06-74a3-4c98-b2c2-278b326ee179](https://github.com/Jeevithha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123623197/8436700a-1149-4c6d-8287-f0a85d62048d)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

