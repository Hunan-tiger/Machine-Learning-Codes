import math
import operator
def creatdataset():
    dataset=[[0, 0, 0, 0, 'no'],#数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [0, 1, 0, 1, 'yes'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 1, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 1, 'no'],
            [2, 1, 0, 0, 'no']]
    label=['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataset,label
    
def calculateshang(dataset):
    numsize=len(dataset)
    labelcount={}
    for i in dataset:
        label=i[-1]
        if label not in labelcount.keys():
            labelcount[label]=0
        labelcount[label]+=1
    shang=0.0
    for key in labelcount:
        p=labelcount[key]/numsize
        shang=-p*math.log(p,2)+shang
    return shang

def separatedataset(dataset,i,value):
    sedataset=[]
    for j in dataset:
        if j[i]==value:
            rej=j[:i]
            rej.extend(j[i+1:])
            sedataset.append(rej)
    return sedataset
    
def calculateinforgain(dataset):
    numsize=len(dataset[0])-1
    xxzy=0.0
    bestfeature=-1
    for i in range(numsize):
        requireshang=0.0
        list01=[j[i] for j in dataset]
        uniquevalue=set(list01)
        for value in uniquevalue:
            setdataset=separatedataset(dataset,i,value)
            p=len(setdataset)/float(len(dataset))
            requireshang+=p*calculateshang(setdataset)
        t=calculateshang(dataset)-requireshang
        if(t>xxzy):
            xxzy=t
            bestfeature=i
    return bestfeature

def bestclass(list02):
    classcount={}
    for cla in list02:
        if cla not in classcount.keys():
            classcount[cla]=0
        classcount[cla]+=1
    sortclass=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)#降序排列
    return sortclass[0][0]
    
def creatTree(dataset,label,bestlabels):
    list02=[example[-1] for example in dataset]
    if list02.count(list02[0])==len(list02):#只有一个特征
        return list02[0]
    if len(dataset[0])==1:#没有特征可选
        return bestclass(list02)
    bestfeature=calculateinforgain(dataset)#最大信息增益特征索引
    bestlabel=label[bestfeature]
    bestlabels.append(bestlabel)
    designTree={bestlabel:{}}#构建决策树
    del label[bestfeature]#删掉该特征
    Values=[example[bestfeature] for example in dataset]
    uniquirevalue=set(Values)#去掉重复特征
    for value in uniquirevalue:
        designTree[bestlabel][value]=creatTree(separatedataset(dataset,bestfeature,value),label,bestlabels)#迭代
    return designTree
    
if __name__=='__main__':
    bestlabels=[]
    dataset,label=creatdataset()
    Tree=creatTree(dataset,label,bestlabels)
    print(Tree)