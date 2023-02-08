import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as pl
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import re
data = pd.read_csv('cars.csv')
print(data.isna().sum())
data['mileage'].fillna(data['mileage'].mean(),inplace=True)
data['engine'].fillna(data['engine'].mean(),inplace=True)
data['max_power'].fillna(data['max_power'].mean(),inplace=True)
data['seats'].fillna(data['seats'].mean(),inplace=True)
data.drop(['torque','owner','seller_type'],axis=1,inplace=True)
typex = pd.get_dummies(data['transmission'],drop_first=True)
data['Transmission']= typex

plt.figure(figsize=(7, 5))
ax = sns.scatterplot(x='mileage',y='max_power', hue='Transmission', data=data, style='Transmission', s=90)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[1:], ['accept', 'Refuse'])
plt.title('Scatter plot of training data')
plt.show()
def sigmoid(z):
    z = np.array(z)
    return 1 / (1+np.exp(-z))
import matplotlib.pyplot as plt

#matplotlib inline
z = np.linspace(-10, 10, 100)
sig = sigmoid(z)
plt.figure(figsize=(9, 3))
plt.plot([-10, 10], [0, 0], "k-")
plt.plot([-10, 10], [0.5, 0.5], "k:")
plt.plot([-10, 10], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(z, sig, "b-", linewidth=2)
plt.xlabel("z")
plt.axis([-10, 10, -0.1, 1.1])
plt.show()

def cost_function(theta, X, y):
    m = y.shape[0]
    theta = theta[:, np.newaxis] #trick to make numpy minimize work
    h = sigmoid(X.dot(theta))
    J = (1/m) * (-y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1-h)))

    diff_hy = h - y
    grad = (1/m) * diff_hy.T.dot(X)

    return J, grad
m = data.shape[0]
X = np.hstack((np.ones((m,1)),data[['mileage', 'max_power']].values))
y = np.array(data.Transmission.values).reshape(-1,1)
initial_theta = np.zeros(shape=(X.shape[1]))
cost, grad = cost_function(initial_theta, X, y)
print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros):')
print(grad.T)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628')
test_theta = np.array([-24, 0.2, 0.2])
[cost, grad] = cost_function(test_theta, X, y)

print('Cost at test theta:', cost)
print('Expected cost (approx): 0.218')
print('Gradient at test theta:')
print(grad.T)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647')

def optimize_theta(X, y, initial_theta):
    opt_results = opt.minimize(cost_function, initial_theta, args=(X, y), method='TNC',jac=True, options={'maxiter':400})
    return opt_results['x'], opt_results['fun']

opt_theta, cost = optimize_theta(X, y, initial_theta)
print('Cost at theta found by fminunc:', cost)
print('Expected cost (approx): 0.203')
print('theta:\n', opt_theta.reshape(-1,1))
print('Expected theta (approx):')
print(' -25.161\n 0.206\n 0.201')

plt.figure(figsize=(7,5))

ax = sns.scatterplot(x='mileage', y='max_power', hue='Transmission', data=data, style='Transmission', s=80)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[1:], ['Accept', 'Refuse'])
plt.title('Training data with decision boundary')

plot_x = np.array(ax.get_xlim())
plot_y = (-1/opt_theta[2]*(opt_theta[1]*plot_x + opt_theta[0]))
plt.plot(plot_x, plot_y, '-', c="red")

plt.show()
prob = sigmoid(np.array([1, 45, 85]).dot(opt_theta))
print('Expected value: 0.775 +/- 0.002')
def predict(X, theta):
    y_pred = [1 if sigmoid(X[i, :].dot(theta)) >= 0.5 else 0 for i in range(0, X.shape[0])]
    return y_pred
X = np.hstack((np.ones((m,1)),data[['mileage', 'max_power']].values))

y_pred_prob = predict(X, opt_theta)
print(f'Train accuracy: {np.mean(y_pred_prob == data.Transmission.values) * 100}')
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


X = data[['engine', 'max_power']]
y = data['Transmission']

#split the dataset into training (70%) and testing (30%) sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#instantiate the model
log_regression = LogisticRegression()

#fit the model using the training data
log_regression.fit(X_train,y_train)




y_pred_proba = log_regression.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




