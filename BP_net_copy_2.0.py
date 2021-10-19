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
    
def unaccuracy(data_te,w_mid,w_out,hidden_num):
    ERR_te=[]
    X=data_te[0:1,1:]
    in_mid=np.zeros(len(X)+1)
    in_mid[0]=-1
    in_out=np.zeros((hidden_num+1))
    in_out[0]=-1
    for t in range(len(data_te)):
        in_mid[1]=data_te[t][1]
        real_te=data_te[t][0]
        for n in range(hidden_num):
            in_out[n+1]=sigmoid(sum(in_mid*w_mid[:,n]))
            
        predict_te=sum(in_out*w_out)
        ERR_te.append(abs(real_te-predict_te))
    ERR_te=np.array(ERR_te)
    e_mean=ERR_te.mean()
    return e_mean
    
def BP_net(data,data_te,train_num,hidden_num):
    X=data[0:1,1:]
    #隐层输入
    w_mid=np.random.rand(len(X)+1,hidden_num)#输入点*隐层点
    in_mid=np.zeros(len(X)+1)
    in_mid[0]=-1#-1乘以一个值为阈值
    #输出层输入
    w_out=np.random.rand(hidden_num+1)
    in_out=np.zeros((hidden_num+1))
    in_out[0]=-1
    
    delta_w_mid=np.zeros((2,hidden_num))
    delta_w_out=np.zeros((hidden_num+1))
    
    yita=0.05#学习率
    ERR=[]#记录平均误差
    errs=1000.0
    nums=0
    for num in range(train_num):
        err=[]
        for i in range(len(data)):
            in_mid[1:]=data[i][1:]
            real=data[i][0]

            for j in range(hidden_num):#按隐层点依次输出
                in_out[j+1]=sigmoid(sum(in_mid*w_mid[:,j]))
            # print(in_out)
            # print(w_out)
            predict=sum(in_out*w_out)
            #err.append(abs(real-predict))
            
            #调整输出层权值与阈值
            delta_w_out=yita*in_out*(real-predict)
            delta_w_out[0]=-yita*(real-predict)
            w_out=w_out+delta_w_out
            #print(delta_w_out)
            #调整隐层权值与阈值
            e=np.zeros(hidden_num)
            for m in range(hidden_num):
                e[m]=in_out[m+1]*(1-in_out[m+1])*w_out[m+1]*(real-predict)
                delta_w_mid[:,m]=yita*e[m]*in_mid[1:]
                delta_w_mid[0,m]=-yita*e[m]
                
            w_mid=w_mid+delta_w_mid
        
        #早停
        # temp=unaccuracy(data_te,w_mid,w_out,hidden_num)
        # if temp<=errs:
            # errs=temp
            # nums=0
        # else:
            # nums+=1
        # if nums>50:
            # break
        print('训练次数:'+str(num+1))
    ERR_te=[]
    predict_te=[]
    for t in range(len(data_te)):
        in_mid[1]=data_te[t][1]
        real_te=data_te[t][0]
        
        for n in range(hidden_num):
            in_out[n+1]=sigmoid(sum(in_mid*w_mid[:,n]))
        predict_te.append(sum(in_out*w_out))
        ERR_te.append((real_te-sum(in_out*w_out))**2)
    plt.plot(data_te[:,1],predict_te,c='r',label='预测')
    plt.scatter(data_te[:,1],data_te[:,0],c='g',s=5,label='数据点')
   # plt.scatter(data[:,1],data[:,0],c='b',s=5,label='训练点')
    plt.rcParams['font.sans-serif'] = ['SimHei']#显示中文
    plt.rcParams['axes.unicode_minus']=False#显示负数
    plt.title('BP_net')
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.plot(ERR_te,c='r',label='误差')
    plt.rcParams['font.sans-serif'] = ['SimHei']#显示中文
    plt.rcParams['axes.unicode_minus']=False#显示负数
    plt.grid()
    plt.legend()
    plt.show()
    
if __name__=='__main__':
    data=dataset('C:\\Users\\Dell\\Desktop\\regression\\sinc_train.txt')
    data_te=dataset('C:\\Users\\Dell\\Desktop\\regression\\sinc_test.txt')
    #ERR=BP_net(data,1000)
    BP_net(data,data_te,3000,11)