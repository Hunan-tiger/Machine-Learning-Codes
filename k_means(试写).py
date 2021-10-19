import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def creatdataset():
    dataset=[[2,3],[4,2],[1,3],[1,4],[5,2],[7,1],[5,1],[1,4],[3,9],[2,8]]
    return dataset
    
def caldistance(dataset,centers,k):
    distancelist=[]
    for data in dataset:
        a=np.tile(data,(k,1))-centers
        b=(a**2).sum(1)
        distance=b**0.5
        distancelist.append(distance)
    distancelist=np.array(distancelist)
    return distancelist
    
def calcenter(dataset,centers,k):
    distances=caldistance(dataset,centers,k)
    mindistnodes=np.argmin(distances,axis=1)
    newcenters=pd.DataFrame(dataset).groupby(mindistnodes).mean()
    #DataFrame(dataSet)对dataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newcenters=newcenters.values
    changed=newcenters-centers
    return changed,newcenters
    
def k_means(dataset,k):
    centers=random.sample(dataset,k)#随机选取k个质心
    changed,newcenters=calcenter(dataset,centers,k)
    i=0
    while np.any(changed)!=0 or i<=99:#np.any判断矩阵中是否有一个元素不为0 若有则返回True 全为0返回False
        changed,newcenters=calcenter(dataset,newcenters,k)
        i+=1
    return newcenters

if __name__=='__main__':
    dataset=creatdataset()
    centers=k_means(dataset,2)
    for i in range(len(dataset)):
        plt.scatter(dataset[i][0],dataset[i][1],c='g',marker='o',label='数据点')
    for j in range(len(centers)):
        plt.scatter(centers[j][0],centers[j][1],c='r',marker='*',label='质心')
    plt.rcParams['font.sans-serif'] = ['SimHei']#显示中文
    plt.legend()
    plt.show()