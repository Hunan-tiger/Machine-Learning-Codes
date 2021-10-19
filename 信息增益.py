import math
def creatdataset():
   dataset=[[0, 0, 0, 0, 'no'],#数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
   labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
   return dataset, labels
    
def calculateshang(dataset):
    numsize=len(dataset)#数据集行数
    labelcount={}#存储各标签的次数
    for i in dataset:
        label=i[-1]#提取标签
        if label not in labelcount.keys():
            labelcount[label]=0
        labelcount[label]+=1
    shang=0.0#经验熵
    for key in labelcount:#公式计算经验熵
        p=labelcount[key]/numsize
        shang-=p*math.log(p,2)
    return shang

def separatedataset(dataset,i,value):#分离数据集从而计算信息增益
    sedataset=[]
    for j in dataset:
        if j[i]==value:
            rej=j[:]
            rej=j[:i]
            rej.extend(j[i+1:])
            sedataset.append(rej)#将第i个特征按不同的特征值分若干个分离数据集
    return sedataset

def calcxxzyandbestfeature(dataset):
    lie_size=len(dataset[0])-1#特征个数
    experienceshang=calculateshang(dataset)
    xxzy=0.0#信息增益
    bestfeature=-1#最优特征索引值 
    for i in range(lie_size):#i个特征
        experequireshang=0.0#经验条件熵
        list01=[example[i] for example in dataset]#将dataSet中的数据先按行依次放入example中，然后取得example中的example[i]元素，放入列表list01中
        uniquevalues=set(list01)#set集合元素不重复
        for value in uniquevalues:
            sepaset=separatedataset(dataset,i,value)
            p=len(sepaset)/float(len(dataset))
            experequireshang+=p*calculateshang(sepaset)
        t=experienceshang-experequireshang
        print("第%d个特征的信息增益为%f"%(i,t))
        if(t>xxzy):
            xxzy=t
            bestfeature=i
    return xxzy,bestfeature

if __name__=='__main__':
    dataset,labels=creatdataset()
    xxzy,bestfeature=calcxxzyandbestfeature(dataset)
    print("最优特征索引值为："+str(xxzy)+","+"该特征的信息增益为："+str(bestfeature))