        self.n_samples, self.n_features = X.shape
        # 计算类别的先验联合概率
        Pypa = {}
        # 计算联合概率的的条件概率
        Pxypa = {}
        yset = np.unique(y)
        # 第一层是不同的分类
        for yi in yset:
            Pypa[yi] = {}; Pxypa[yi] = {}
            
            # 第二层是不同的超父属性，如果是连续值则，不能当作超父，离散值当作超父属性 
            for paIdx in range(self.n_features):
                if columnsMark[paIdx] == 1:
                    continue
                Pypa[yi][paIdx] = {}; Pxypa[yi][paIdx] = {}
                paset = np.unique(X[:, paIdx])
                
                # 第三层是不同的超父属性的属性值，分离出来对应的Xarr，和yarr
                for pai in paset:
                    yi_pai_idx = np.nonzero((X[:,paIdx]==pai)&(y==yi))#返回不是0的元素的索引值
                    
#                    if paIdx==2 and pai==1:
#                        print(yi, '\n', yi_pai_idx)
                    
                    yarr = y[yi_pai_idx]
                    ## 保存类别的先验联合概率
                    Pypa[yi][paIdx][pai] = self.__calyproba(yarr, self.n_samples, len(yset), len(paset))
                    Pxypa[yi][paIdx][pai] = {}
                    
                    # 第四层是不同的其他特征，若是超父属性则跳过，离散归离散统计，连续归连续统计
                    for xiIdx in range(self.n_features):
                        if xiIdx == paIdx:
                            continue
                        allxiset = np.unique(X[:, xiIdx])
                        Xarr = X[yi_pai_idx, xiIdx].flatten()#默认按行降维到一维，只适用数组
                        if columnsMark[xiIdx] == 0:
                            ## 保存离散特征的条件概率
                            Pxypa[yi][paIdx][pai][xiIdx] = self.__categorytrain(Xarr, allxiset)
                        else:
                            ## 保存连续特征的条件概率
                            Pxypa[yi][paIdx][pai][xiIdx] = self.__continuoustrain(Xarr)
                        
#                        if xiIdx == 4 and paIdx==2 and pai==1:
#                            print(Xarr)
                        
        print('P(y,pa)训练完毕!')
        print('P(x|y,pa)训练完毕!')
        self.yProba = Pypa
        self.xyProba = Pxypa
        self.trainSet = X
        self.trainLabel = y
        self.columnsMark = columnsMark        
        return
    
    
    # 计算离散特征的条件概率
    def __categorytrain(self, Xarr, xiset):
        pxypa = {}
        for xivalue in xiset:
            pxypa[xivalue] = {}
            pxypa[xivalue]['count'] = sum(Xarr==xivalue) + self.ls
            pxypa[xivalue]['ratio'] = self.classifyProba(xivalue, Xarr, len(xiset))
        return pxypa
    
    # 计算连续特征的均值和标准差
    def __continuoustrain(self, Xarr):
        pxypa = (Xarr.mean(), Xarr.std())
        return pxypa
        
    # 计算先验联合概率
    def __calyproba(self, yarr, ysum, ysetsum, pasetsum):
        yproba = {}
        yproba['count'] = len(yarr) + self.ls
        yproba['ratio'] = (len(yarr) + self.ls) / (ysum + ysetsum * pasetsum)
        return yproba
    
    
    # 预测
    def aodepredict(self, X, minSet=0):
        n_samples, n_features = X.shape
        proba = np.zeros((n_samples, len(self.yProba)))
        for i in range(n_samples):
            for idx, (yi, Padict) in enumerate(self.yProba.items()):
                sumvalue = 0.
                for paIdx, Pavaluedict in Padict.items():
                    subvalue = 1.
                    pavalue = X[i, paIdx]
                    Statsdict = Pavaluedict[pavalue]
                    if Statsdict['count'] <= minSet:
                        continue
                    Pypa = Statsdict['ratio']
                    subvalue *= Pypa
                    Pxypadict = self.xyProba[yi][paIdx][pavalue]
                    for xiIdx, xiparams in Pxypadict.items():
                        xi = X[i, xiIdx]
                        if isinstance(xiparams, dict):
                            Pxypa = xiparams[xi]['ratio']
                        else:
                            if np.isnan(xiparams[0]) or np.isnan(xiparams[1]):
                                Pxypa = 1.0e-5
                            else:
                                miu = xiparams[0]; sigma = xiparams[1] + 1.0e-5
                                Pxypa = np.exp(-(xi-miu)**2/(2*sigma**2))/(np.power(2*np.pi, 0.5)*sigma) + 1.0e-5
                        subvalue *= Pxypa
                    sumvalue += subvalue
                proba[i, idx] = sumvalue
        return proba
        
dataSet = [
            ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
            ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
            ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
            ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
            ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
            ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
            ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
            ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
            ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
            ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
            ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
            ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
            ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
        ]
    #特征值列表
    labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感']