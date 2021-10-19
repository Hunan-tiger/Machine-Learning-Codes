import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dataset(address):
    dataset=np.loadtxt(address,delimiter=',')
    return dataset
    
def BP_net(data,data_te,train_num):
    yin_num=round(1+(1*(1+2))**0.5)#隐层结点个数
    #隐层输入
    w_mid=np.random.rand(2,yin_num)#输入点*隐层点
    in_mid=np.array([-1,0.0])#-1乘以一个值为阈值
    #输出层输入
    w_out=np.random.rand(yin_num+1)
    in_out=np.array([-1,0.0,0.0,0.0])
    
    delta_w_mid=np.zeros((2,yin_num))
    delta_w_out=np.zeros((yin_num+1))
    
    yita=0.1#学习率
    ERR=[]#记录平均误差
    for num in range(train_num):
        err=[]
        for i in range(len(data)):
            in_mid[1]=data[i][0]
            real=data[i][1]

            for j in range(yin_num):#按隐层点依次输出
                in_out[j+1]=sigmoid(sum(in_mid*w_mid[:,j]))
            predict=sigmoid(sum(in_out*w_out))
            err.append(abs(real-predict))
            #调整输出层权值与阈值
            delta_w_out=yita*in_out*predict*(1-predict)*(real-predict)
            delta_w_out[0]=-yita*predict*(1-predict)*(real-predict)
            
            w_out+=delta_w_out
            
            #调整隐层权值与阈值
            e=np.zeros(yin_num)
            for m in range(yin_num):
                e[m]=in_out[m]*(1-in_out[m])*w_out[m+1]*predict*(1-predict)*(real-predict)
                delta_w_mid[:,m]=yita*e[m]*in_mid
                delta_w_mid[0,m]=-yita*e[m]
                
            w_mid+=delta_w_mid
        err=np.array(err)
        ERR.append(err.mean())#平均误差
        print('训练次数：'+str(num))
    plt.plot(ERR,c='r',label='训练平均误差')
    plt.rcParams['font.sans-serif'] = ['SimHei']#显示中文
    plt.rcParams['axes.unicode_minus']=False#显示负数
    plt.title('训练平均误差趋势')
    plt.grid()
    plt.legend()
    plt.show()
    
    predict_te=[]
    for t in range(len(data_te)):
        in_mid[1]=data_te[t][0]
        real_te=data_te[t][1]
        
        for n in range(yin_num):
            in_out[n+1]=sigmoid(sum(in_mid*w_mid[:,n]))
        predict_te.append(sigmoid(sum(in_out*w_out)))
        # ERR_te.append(abs(real_te-predict_te))
    plt.plot(predict_te,c='r',label='验证平均误差')
    plt.scatter(data_te[:,0],data[:,1],c='g',label='数据点')
    plt.rcParams['font.sans-serif'] = ['SimHei']#显示中文
    plt.rcParams['axes.unicode_minus']=False#显示负数
    plt.title('验证平均误差趋势')
    plt.grid()
    plt.legend()
    plt.show()
    
if __name__=='__main__':
    data=dataset('C:\\Users\\Dell\\Desktop\\regression\\sinc_train.txt')
    data01=dataset('C:\\Users\\Dell\\Desktop\\regression\\sinc_test.txt')
    print(data01[:0])
    #ERR=BP_net(data,1000)
    BP_net(data,data01,10)