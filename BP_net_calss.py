import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    if x>=0:
        return 1/(1+np.exp(-x))
    else:
        return np.exp(x)/(1+np.exp(x))
        
def dataset(address):
    dataset=np.loadtxt(address,delimiter=',')
    return dataset

def BP_net(data,data_te,train_num,hidden_num):
    #隐层输入
    X=data[0,1:]
    w_mid=np.random.randn(len(X)+1,hidden_num)#输入点*隐层点
    in_mid=np.zeros(len(X)+1)
    in_mid[0]=-1#-1乘以一个值为阈值
    #输出层输入
    w_out=np.random.randn(hidden_num+1)
    in_out=np.zeros((hidden_num+1))
    in_out[0]=-1
    
    delta_w_mid=np.zeros((len(X)+1,hidden_num))
    delta_w_out=np.zeros((hidden_num+1))
    
    yita=0.05#学习率
    ERR=[]#记录平均误差
    errs=1000.0
    nums=0
    for num in range(train_num):
        err=[]
        print('训练次数:'+str(num))
        for i in range(len(data)):
            in_mid[1:]=data[i][1:]
            real=data[i][0]
            
            for j in range(hidden_num):#按隐层点依次输出
                in_out[j+1]=sigmoid(sum(in_mid*w_mid[:,j]))
            predict=sigmoid(sum(in_out*w_out))
            #err.append(abs(real-predict))
            
            #调整输出层权值与阈值
            delta_w_out=yita*in_out*predict*(1-predict)*(real-predict)
            delta_w_out[0]=-yita*predict*(1-predict)*(real-predict)
            w_out=w_out+delta_w_out
            
            #调整隐层权值与阈值
            e=np.zeros(hidden_num)
            for m in range(hidden_num):
                e[m]=in_out[m]*(1-in_out[m])*w_out[m+1]*predict*(1-predict)*(real-predict)
                delta_w_mid[:,m]=yita*e[m]*in_mid
                delta_w_mid[0,m]=-yita*e[m]
            w_mid=w_mid+delta_w_mid
    return w_mid,w_out

if __name__=='__main__':
    data=dataset('C:\\Users\\Dell\\Desktop\\classification\\diabetes_train.txt')
    data_te=dataset('C:\\Users\\Dell\\Desktop\\classification\\diabetes_test.txt')
    #ERR=BP_net(data,1000)
   
    ERR_te=[]
    predict_te=[]
    ps=[(n/100)for n in range(101)]
    TPR=[]
    FPR=[]
    a=-1
    b=0
    AUC=0.0
    hidden_num=64
    X=data[0,1:]
    w_mid,w_out=BP_net(data,data_te,500,64)
    for p in ps:
        print(p)
        in_mid=np.zeros(len(X)+1)
        in_mid[0]=-1#-1乘以一个值为阈值
        in_out=np.zeros((hidden_num+1))
        in_out[0]=-1
        
        for t in range(len(data_te)):
            in_mid[1:]=data_te[t][1:]
            for n in range(hidden_num):
                in_out[n+1]=sigmoid(sum(in_mid*w_mid[:,n]))
            if sigmoid(sum(in_out*w_out))>=p:
                predict_te.append(1)
            else:
                predict_te.append(0)
        tp=0
        fp=0
        tn=0
        fn=0
        for i in range(len(predict_te)):
            if predict_te[i]==data_te[i,0]:
                if predict_te[i]==1:
                    tp+=1
                else:
                    tn+=1
            else:
                if predict_te[i]==1:
                    fp+=1
                else: 
                    fn+=1
        del predict_te[:]
        tpr=tp/(tp+fn)
        fpr=fp/(fp+tn)
        AUC+=(a+tpr)*(b-fpr)/2
        TPR.append(tpr)
        FPR.append(fpr)
        a=tpr
        b=fpr
        #ERR_te.append(abs(real_te-sigmoid(sum(in_out*w_out))))
    #plt.plot(data_te[:,1],predict_te,c='r',label='预测')
    #plt.scatter(data_te[:,1],data_te[:,0],c='g',s=5,label='数据点')
    print('AUC='+str(AUC))
    plt.plot(FPR,TPR,'r',label='ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    x=[(n/10) for n in range(11)]
    y=[(n/10) for n in range(11)]
    plt.xticks(x)
    plt.yticks(y)
    plt.rcParams['font.sans-serif'] = ['SimHei']#显示中文
    plt.rcParams['axes.unicode_minus']=False#显示负数
    plt.title('BP_net_calss')
    plt.legend()
    plt.grid()
    plt.show()