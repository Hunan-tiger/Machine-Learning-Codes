#判断一个视频是否属于幽默类还是实用类
import numpy as np
import operator
from matplotlib import pyplot as plt
def traindataset():
    datagroup=np.loadtxt('C:\\Users\\Dell\Desktop\\classification\\diabetes_train.txt',dtype=float,delimiter=',')
    dataset=datagroup[:,1:]
    label=datagroup[:,0]
    return dataset,label
    
def testdataset():
    datagroup=np.loadtxt('C:\\Users\\Dell\Desktop\\classification\\diabetes_test.txt',dtype=float,delimiter=',')
    dataset=datagroup[:,1:]
    label=datagroup[:,0]
    return dataset,label
    
def K_classify(test,datagroup,label,k,p):#p-阈值
    datasize=datagroup.shape[0]#计算已知数据的行数
    test0=np.tile(test,(datasize,1))-datagroup#将测试集与已知数据形式相同，再相减
    distance0=(test0**2).sum(1)#平方和
    distance=distance0**0.5#开方算欧氏距离
    listsy=distance0.argsort()#距离从小到大按索引(下标)排序
    classcount={}#创建一个空字典
    num0=0
    num1=0
    for i in range(k):
        label0=label[listsy[i]]
        classcount[label0]=classcount.get(label0,0)+1#计算各类别的次数
        if label0==0:
            num0+=1
        else:
            num1+=1
    nums=num0+num1
    if num1/nums >= p:
        return 1
    else:
        return 0
    
if __name__=='__main__':
    datagroup,label=traindataset()
    test,truelabels=testdataset()
    predict=[]
    Ps=[(n/100) for n in range(101)]#改变阈值
    a=-1
    b=0
    AUC=0.0
    TPR=[]
    FPR=[]
    for p in Ps:
        for i in range(len(test)):
            predict.append(K_classify(test[i],datagroup,label,150,p))
        tp=0
        fp=0
        tn=0
        fn=0
        for j in range(len(test)):
            if predict[j]==truelabels[j]:
                if predict[j]==1:
                    tp+=1
                else:
                    tn+=1
            else:
                if predict[j]==1:
                    fp+=1
                else:
                    fn+=1
        fpr=fp/(fp+tn)
        tpr=tp/(tp+fn)
        print(fpr)
        AUC+=(a+tpr)*(b-fpr)/2#微元法算梯形面积求AUC
        a=tpr
        b=fpr
        TPR.append(tpr)
        FPR.append(fpr)
        del predict[:]
    plt.plot(FPR,TPR)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC曲线')
    plt.rcParams['font.sans-serif'] = ['SimHei']#显示中文
    plt.grid()#网格线
    x=[(n/10) for n in range(11)]
    y=[(n/10) for n in range(11)]
    plt.xticks(x)
    plt.yticks(y)
    print('AUC=',AUC)
    plt.show()
