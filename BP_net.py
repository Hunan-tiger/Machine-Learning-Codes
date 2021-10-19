import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    if x>0:
        return 1/(1+np.exp(-x))
    else :
        return np.exp(x)/(1+np.exp(x))

def dataset(address):
    dataset=np.loadtxt(address,delimiter=',')
    return dataset
    
def unaccuracy(data_te,w_mid,w_out,hidden_num):
    ERR_te=[]
    in_mid=np.array([-1,0.0])
    in_out=np.array([-1,0.0,0.0,0.0])
    for t in range(len(data_te)):
        in_mid[1]=data_te[t][1]
        real_te=data_te[t][0]
        for n in range(hidden_num):
            in_out[n+1]=sigmoid(sum(in_mid*w_mid[:,n]))
        predict_te=sigmoid(sum(in_out*w_out))
        ERR_te.append(abs(real_te-predict_te))
    ERR_te=np.array(ERR_te)
    e_mean=ERR_te.mean()
    return e_mean
    
def BP_net(data,data_te,train_num):
    hidden_num=round(1+(1*(1+2))**0.5)#隐层结点个数
    #隐层输入
    w_mid=np.random.rand(2,hidden_num)#输入点*隐层点
    in_mid=np.array([-1,0.0])#-1乘以一个值为阈值
    #输出层输入
    w_out=np.random.rand(hidden_num+1)
    in_out=np.array([-1,0.0,0.0,0.0])
    
    delta_w_mid=np.zeros((2,hidden_num))
    delta_w_out=np.zeros((hidden_num+1))
    
    yita=0.1#学习率
   #记录平均误差
    nums=0
    errs=1000
    t=0
    err_t=1000.0
    w_mid_best=np.zeros((2,hidden_num))
    w_out_best=np.zeros((hidden_num+1))
    data_size=round(len(data)/10)
    for k in range(9):#k折交叉验证 
        #ERR=[]
        print(k)
        data_k=data[k*data_size:(k+1)*data_size]
        err=[]
        for num in range(train_num):
            
            for i in range(len(data_k)):
                in_mid[1]=data_k[i][1]
                real=data_k[i][0]

                for j in range(hidden_num):#按隐层点依次输出
                    in_out[j+1]=sigmoid(sum(in_mid*w_mid[:,j]))
                predict=sigmoid(sum(in_out*w_out))
                err.append(abs(real-predict))
                #调整输出层权值与阈值
                delta_w_out=yita*in_out*predict*(1-predict)*(real-predict)
                delta_w_out[0]=-yita*predict*(1-predict)*(real-predict)
                
                w_out+=delta_w_out
                
                #调整隐层权值与阈值
                e=np.zeros(hidden_num)
                for m in range(hidden_num):
                    e[m]=in_out[m]*(1-in_out[m])*w_out[m+1]*predict*(1-predict)*(real-predict)
                    delta_w_mid[:,m]=yita*e[m]*in_mid
                    delta_w_mid[0,m]=-yita*e[m]
                    
                w_mid+=delta_w_mid
            # err=np.array(err)
            #ERR.append(err.mean())#平均误差
            # print('训练次数：'+str(num))
        #验证集验证
        data_v=data[9*data_size:]
        for d in range(len(data_v)):
            in_mid[1]=data_v[d][1]
            real_te=data_v[d][0]
            
            for n in range(hidden_num):
                in_out[n+1]=sigmoid(sum(in_mid*w_mid[:,n]))
            predict_te=sigmoid(sum(in_out*w_out))
            err.append(abs(real_te-predict_te))
        err=np.array(err)
        Err=err.mean()
        if Err<err_t:
            w_mid_best=w_mid
            w_out_best=w_out
            err_t=Err
            
    predict_te=[]
    for t in range(len(data_te)):
        in_mid[1]=data_te[t][1]
        real_te=data_te[t][0]
            
        for n in range(hidden_num):
            in_out[n+1]=sigmoid(sum(in_mid*w_mid_best[:,n]))
        predict_te.append(sigmoid(sum(in_out*w_out_best)))
        # ERR_te.append(abs(real_te-predict_te))
        
    plt.plot(data_te[:,1],predict_te,c='r',label='预测')
    plt.rcParams['font.sans-serif'] = ['SimHei']#显示中文
    plt.rcParams['axes.unicode_minus']=False#显示负数
    plt.scatter(data_te[:,1],data_te[:,0],c='g',s=5,label='数据点')
    plt.title('BP_net')
    plt.grid()
    plt.legend()
    plt.show()
    
if __name__=='__main__':
    data=dataset('C:\\Users\\Dell\\Desktop\\regression\\sinc_train.txt')
    data01=dataset('C:\\Users\\Dell\\Desktop\\regression\\sinc_test.txt')
    #ERR=BP_net(data,1000)
    BP_net(data,data01,100)