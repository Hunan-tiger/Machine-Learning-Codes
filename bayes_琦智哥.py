import numpy as np
import random
import math
def data_split(data):#将原数据划分成含有120个样本的训练集，含有30个样本的测试集
    data = data[1:]
    train_ratio = 0.8
    train_size = int(len(data) * train_ratio)
    random.shuffle(data)
    train_set = data[:train_size]
    test_set = data[train_size:]
    return train_set, test_set

def class_separate(data_set):#对数据进行处理，分类并统计
    separated_data = {}
    data_info = {}#表示的是各品种花所对应的数量
    for data in data_set:
        if data[-1] not in separated_data:#创建新字典索引，data和info两个字典都各自有三个索引，即花的品种
            separated_data[data[-1]] = []
            data_info[data[-1]] = 0
        separated_data[data[-1]].append(data)#将所有花分成三类，每种花都会被添加进对应的花类别中
        data_info[data[-1]] += 1
    if 'Species' in separated_data:
        del separated_data['Species']
    if 'Species' in data_info:
        del data_info['Species']
    return separated_data, data_info

def prior_prob(data, data_info):#计算每个类的先验概率
    data_prior_prob = {}
    data_sum = len(data)
    for cla, num in data_info.items():
        data_prior_prob[cla] = num / float(data_sum)
    return data_prior_prob

def mean(data):#求均值
    data = [float(x) for x in data]#字符串转数字
    return sum(data) / len(data)

def var(data):#求方差
    data = [float(x) for x in data]
    mean_data = mean(data)
    var = sum([math.pow((x - mean_data), 2) for x in data]) / float(len(data) - 1)
    return var

def prob_dense(x, mean, var):#由于是样本属性是连续值，所以要用概率密度
    exponent = math.exp(math.pow((float(x) - mean), 2) / (-2 * var))
    p = (1 / math.sqrt(2 * math.pi * var)) * exponent
    return p

def attribute_info(data):#分别计算出四个属性的均值和方差
    dataset = np.delete(data, -1, axis = 1) #删除标签
    result = [(mean(att), var(att)) for att in zip(*dataset)]#zip解开元组，生成四个属性的结果
    return result

def summarize_class(data):
  data_separated, data_info = class_separate(data)
  summarizeClass = {}
  for index, x in data_separated.items():#三种类别对应4种属性，共12组，每组包含对应的均值和方差
      summarizeClass[index] = attribute_info(x)
  return summarizeClass

def cla_prob(test, summarizeClass):#此时summarizeClass就是三个类别的概率密度函数参数，即模型参数，test即测试数据（样本数据)
  prob = {}
  for cla, summary in summarizeClass.items():#cla类别，summary为该类对应的四个属性的（均值，方差）集合
      prob[cla] = 1
      for i in range(len(summary)):#属性个数
          mean, var = summary[i]
          x = test[i]
          p = prob_dense(x, mean, var)#计算离散属性值的条件概率
      prob[cla] *= p #连乘，最后得到这个类关于所有属性的条件概率
  return prob#获得条件概率

def bayesian(input_data, data, data_info):#只需要考虑贝叶斯公式的分子，因为分母都是一样的，即只要考虑先验概率和条件概率
  priorProb = prior_prob(data, data_info)
  data_summary = summarize_class(data)
  classProb = cla_prob(input_data, data_summary)
  result = {}
  for cla, prob in classProb.items():
      # print(type(classProb))
      p = prob * priorProb[cla]
      result[cla] = p
  return max(result, key=result.get)

iris = np.array(np.loadtxt('C:\\Users\\Dell\Desktop\\classification\\diabetes_train.txt', delimiter=",")).tolist()##导入鸢尾花数据，共有150个样本，其中有四个属性，3个类别（0,1,2），每个类别都有50个样本
train_set, test_set = data_split(iris)
train_separated, train_info = class_separate(train_set)#获得分类好的数据和每个类别的样本数
correct = 0
for x in test_set:
  input_data = x[:-1]
  label = x[-1]
  result = bayesian(input_data, train_set, train_info)
  if result == label:
      correct += 1
print(correct / len(test_set))
