import numpy as np
import matplotlib.pyplot as plt
import math
#高斯朴素贝叶斯

def dataset(path):
    dataset=np.loadtxt(path,delimiter=',')
    return dataset

def mean(x):#均值
    return sum(x)/float(len(x))

def var(list1):#方差
    list1=list(list1)
    var1=np.sum([math.pow(x-mean(list1),2) for x in list1])/float(len(list1))
    return var1

def require_p(x,mean,var1):#条件概率
    exp=math.exp(-math.pow(float(x)-mean,2)/(2*var1))
    p_re=exp/math.sqrt(2*math.pi*var1)
    return p_re

def bayes(data_tr,data_te,p):
    #训练
    data=np.array(data_tr[:,1:])
    labels=data_tr[:,0]
    classcount={}
    p_label={}
    n=labels.shape[0]#数据量

    #先验概率：
    for i in labels:
        classcount[i]=classcount.get(i,0)+1
    label=set(labels)
    for i in label:
        p_label[i]=(classcount.get(i,0))/float(n)
        
    row_num=data[0].shape[0]#计算特征数
    
    m=np.zeros((row_num,2))#存储第i个特征第j个类别的均值
    v=np.zeros((row_num,2))#存储第i个特征第j个类别的方差
    #计算高斯参数——均值和方差
    for i in label:
        i=int(i)
        for j in range(row_num):#遍历特征
            #uni[j]=set(data[:,j])#每个特征不重复属性
            temp=[]
            t=0
            for l in data[:,j]:
                if labels[t]==i:#Xi,Yk时参数
                    temp.append(l)
                t+=1
            m[j][i]=mean(temp)
            v[j][i]=var(temp)

    #测试：
    te_labels=[]
    features=data_te[:,1:]
    for j in range(len(features)):#遍历测试数据
        te_p_labels={}
        for i in label:
            i=int(i)
            te_p_labels[i]=p_label[i]
            t=0
            for k in features[j]:
                te_p_labels[i]=te_p_labels[i]*require_p(k,m[t][i],v[t][i])
                t+=1
        #print(j)
        sum1=te_p_labels[0]+te_p_labels[1]#归一化
        te_p_labels[1]=te_p_labels[1]/sum1
        te_p_labels[0]=te_p_labels[0]/sum1
        # for t in te_p_labels.keys():
            # maxlabel=t
            # break
       # for t in te_p_labels.keys():#分母一样，只比较分子
        if te_p_labels[1]>=p:#te_p_labels.get(maxlabel):
            te_labels.append(1)
            #maxlabel=t
        else:
            te_labels.append(0)
    return te_labels

if __name__=='__main__':
    data_tr=dataset('C:\\Users\\Dell\Desktop\\classification\\diabetes_train.txt')
    data_te=dataset('C:\\Users\\Dell\Desktop\\classification\\diabetes_test.txt')
    
    truelabels=data_te[:,0]
    # print(labels)
    # count=0
    # for i in range(len(labels)):
        # if labels[i]==truelabels[i]:
            # count+=1
    # print("准确率："+str(float(count)/len(labels)))
    
    a=-1
    b=0
    AUC=0.0
    TPR=[]
    FPR=[]
    ps=[(n/100) for n in range(101)]
    for p in ps:
        labels=bayes(data_tr,data_te,p)
        tp=0
        tn=0
        fp=0
        fn=0
        for i in range(len(labels)):
            if labels[i]==truelabels[i]:
                if labels[i]==0:
                    tn+=1
                else :
                    tp+=1
            else :
                if labels[i]==0:
                    fn+=1
                else:
                    fp+=1
        fpr=fp/(fp+tn)
        tpr=tp/(tp+fn)
        AUC+=(a+tpr)*(b-fpr)/2
        #print((a+tpr)*(b-fpr)/2)
        a=tpr
        b=fpr
        TPR.append(tpr)
        FPR.append(fpr)
        print(p)
    num=[]
    for i in range(len(data_te)):
        num.append(i)
    
   # plt.title('bayes')
    plt.title('AUC='+str(AUC))
    plt.plot(FPR,TPR)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    x=[x/10 for x in range(11)]
    y=[y/10 for y in range(11)]
    plt.xticks(x)
    plt.yticks(y)
    #print('AUC='+str(AUC))
   # plt.scatter(num,labels,c='r',s=5,label='预测值')
   # plt.scatter(num,truelabels,c='g',s=5,label='实际值')
    plt.rcParams['font.sans-serif'] = ['SimHei']#显示中文
    plt.rcParams['axes.unicode_minus']=False#显示负数
    plt.grid()
    #plt.legend()
    plt.show()