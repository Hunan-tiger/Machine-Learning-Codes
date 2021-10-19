import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.metrics import classification_report
from copy import deepcopy
import sys

def sigmoid(z):
 #   print('cost'+str(z)+'='+str(1 / (1 + np.exp(-z))))
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return np.multiply( sigmoid(z) , 1 - sigmoid(z) )

#全局的theta定义的是 X*theta 的方式

def forward_prop(X,theta1,theta2): #前向传播
    X = np.matrix(X)
    doe = np.ones(X.shape[0])
    X = np.c_[ doe , X ]                             #注意这里是''[]''
    z1 = X * theta1  ;  a1 = sigmoid(z1)
    a1 = np.c_[doe , a1]
    z2 = a1 * theta2 ;  a2 = np.matrix(z2)
    return z1,a1,z2,a2

def cost_reg(X,y,theta1,theta2,LearningRate):
    X = np.matrix(X) ; m=X.shape[0]
    z1,a1,z2,a2 = forward_prop(X,theta1,theta2)

    ans = float(np.sum(np.power(a2-y,2))) / (2*m)
    ans += float(LearningRate) / (2*m) * ( np.sum( np.power(theta1[1:,:],2) )  +  np.sum( np.power(theta2[1:,:],2) ) )
    return ans

def back_prop(X,y,theta1,theta2,LearningRate):
    X = np.matrix(X) ; y = np.matrix(y)
    z1,a1,z2,a2 = forward_prop(X,theta1,theta2)

    delta1 = np.zeros(theta1.shape) ; delta2 = np.zeros(theta2.shape)
    m = X.shape[0]
    for i in range(m):
        _z1 = np.c_[1 , z1[i,:] ] ; _a1 = a1[i,:]
        _z2 = z2[i,:]  ; _a2 = a2[i,:]
        _a0 = np.c_[1 , X[i,:] ]

        u2 =  _a2 - y[i,:]
        u1 = np.multiply( u2 * theta2.T , sigmoid_gradient(_z1) )           #注意输出端没有扩展
        delta2 += _a1.T * u2
        delta1 += _a0.T * u1[:,1:]                                           #去除扩展项

    delta1[1:,:] += float(LearningRate) * theta1[1:,:]                         #注意0项不加！！！
    delta2[1:,:] += float(LearningRate) * theta2[1:,:]

    delta1/=m ; delta2/=m
    return delta1,delta2




#################################################
path ='C:\\Users\\Dell\\Desktop\\regression\\sinc_train.txt'
train = pd.read_csv(path , header=None )
y = train.iloc[:,0:1]
X = train.iloc[:,1:]
X = np.matrix(X.values) ; y =np.matrix(y.values)
print(X);print(y)


X = X
m=X.shape[0]


theta1 = (np.random.rand(X.shape[1]+1,64)-0.5) *2
theta2 = (np.random.rand(65,1)-0.5) *2.5
z1,a1,z2,yy = forward_prop(X,theta1,theta2)

'''
print(z1)
print(z1.min(),z1.max())
print(yy)
'''

alpha=0.1 ; LearningRate=0.01 ; lim=0.0006 ; step=1
pre = cost_reg(X,y,theta1,theta2,LearningRate)
while (True):
    d1,d2 = back_prop(X,y,theta1,theta2,LearningRate)
    theta1 -= d1 * alpha    ;   theta2 -= d2 * alpha
    now = cost_reg(X,y,theta1,theta2,LearningRate)
#    if (pre - now < lim) : break
    if (step > 100) : break
    if (step%1==0) : print(step,now)
    pre = now ; step+=1


xx = np.linspace(X.min(),X.max(),500)
xx = np.reshape(xx,(500,1)) ; xx=np.matrix(xx)
z1,a1,z2,yy = forward_prop(xx,theta1,theta2)
yy.flatten() ; xx=xx ; xx.flatten()

X = X
fig,ax = plt.subplots(figsize=(12,8))
print(xx.shape);print(yy.shape)
x_plot=np.array(xx) ; y_plot=np.array(yy)
ax.plot(x_plot,y_plot,c='r',label='pred')
xx=np.array(X) ; yy=np.array(y)
ax.scatter(xx,yy,s=30)
plt.show()

sys.exit()
z1,a1,z2,y_pred = forward_prop(X,theta1,theta2)
y_pred = np.array(np.argmax(y_pred,axis=1))
for i in range(m):
    if (y_pred[i]==0):y_pred[i]=10
y = np.array(data['y'])
print(classification_report(y, y_pred))


################################################
sys.exit()

xx=np.array(X) ; yy=np.array(y)
fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(xx,yy,s=30)
plt.show()
