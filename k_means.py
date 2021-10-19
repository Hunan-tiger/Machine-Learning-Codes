# dataSet样本点,k 簇的个数
# disMeas距离量度，默认为欧几里得距离
# createCent,初始点的选取
import numpy
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = dataSet.shape[0] #样本数
    clusterAssment = mat(zeros((m,2))) #m*2的矩阵                   
    centroids = createCent(dataSet, k) #初始化k个中心
    clusterChanged = True             
    while clusterChanged:      #当聚类不再变化
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k): #找到最近的质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            # 第1列为所属质心，第2列为距离
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)

        # 更改质心位置
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0) 
    return centroids, clusterAssment