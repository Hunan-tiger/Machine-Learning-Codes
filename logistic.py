import numpy as np
import matplotlib.pyplot as plt

def traindataset():
    dataset=np.loadtxt('C:\\Users\\Dell\\Desktop\\regression\\sinc_train.txt',delimiter=',')
    x=dataset[:,0]
    xmat=np.mat(x)#1*5000
    y=dataset[:,1]
    ymat=np.mat(y).T#5000*1
    return xmat,ymat
    
def w_train(xmat,ymat,alpha,iternum):
    w=np.random.randn(1,1)
    x=np.asarray(xmat)
    for i in range(iternum):
        if w*xmat.all()>0:
            p=1/(1+np.exp(-w*xmat))
        else :
            p=np.exp(w*xmat)/(1+np.exp(w*xmat))
        dw=xmat*(ymat-p.T)
        w+=alpha*dw
        if dw==0:#不再下降
            return w[0][0]
    return w[0][0]
    
def testdataset():
    dataset=np.loadtxt('C:\\Users\\Dell\\Desktop\\regression\\sinc_train.txt',delimiter=',')
    x=dataset[:,0]
    y=dataset[:,1]
    return x,y
    
if __name__=='__main__':
    xmat,ymat=traindataset()
    w=w_train(xmat,ymat,0.001,10000)
    print('w=',w)
    x1,y_truth=testdataset()
    y1=[]
    for i in range(len(x1)):
        if w*x1[i]>0:
            y1.append(1/(1+np.exp(w*x1[i])))
        else :
            y1.append(np.exp(w*x1[i])/(1+np.exp(w*x1[i])))
    plt.rcParams['font.sans-serif'] = ['SimHei']#显示中文
    plt.rcParams['axes.unicode_minus']=False#显示负数
    plt.plot(x1,y1,'r.',label='决策边界',linewidth=1)
    plt.scatter(x1,y_truth,c='g',s=5,label='数据点')
    plt.grid()
    plt.legend()
    plt.show()
