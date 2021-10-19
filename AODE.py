import numpy as np
import matplotlib.pyplot as plt

def dataset():
    dataSet = [
            ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
            ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
            ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
            ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
            ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
            ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
            ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
            ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
            ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
            ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
            ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
            ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],#test
            ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']#test
        ]
    #特征
    labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感']
    return dataSet,labels
    
def beforep(D,Dcx,N,Ni):#先验概率
    p=(Dcx+1)/(D+N*Ni)
    return p

def afterp(Dcij,Dcx,Nj):#后验概率
    p=(Dcij+1)/(Dcx+Nj)
    return p

def AODE(data_te):
    #训练
    data,labels=dataset()
    data=np.array(data[2:])
    class1=set(data[:,-1])
    data_te=np.array(data_te)
    D=data.shape[0]#数据量
    N=len(set(class1))
    test=data_te[:,:-1]
    turelabels=data_te[:,-1]
    for m in range(len(test)):
        p={}
        for j in class1:
            p[j]=0
            for i in range(len(labels)):
                Ni=len(set(data[:,i]))#第i个属性的取值数
                Dcxi=data[(data[:,i]==test[m][i])&(data[:,-1]==j)]#c类别Xi的个数（|Dxi|>0）
                pcxi=(len(Dcxi)+1)/(D+N*Ni)
                pxjcxi=1
                for z in range(len(labels)):
                    if z==i:
                        continue
                    Nj=len(set(data[:,z]))
                    Dcxixj=data[(data[:,i]==test[m][i])&(data[:,-1]==j)&(data[:,z]==test[m][z])]
                    pcij=(len(Dcxixj)+1)/(len(Dcxi)+Nj)
                    pxjcxi=pcij*pxjcxi
            p[j]+=pcxi*pxjcxi
        P=max(p,key=p.get)
        print(test[m])
        print('预测:'+P)
        print('实际:'+turelabels[m])
        
if __name__=='__main__':
    data_te,i=dataset()
    data_te=data_te[-2:]
    AODE(data_te)