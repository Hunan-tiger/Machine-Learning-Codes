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
        p=labelcount[key]/numsize#0<=p<=1
        shang-=p*math.log(p,2)
    return shang

if __name__=='__main__':
    dataset,labels=creatdataset()
    print(calculateshang(dataset))