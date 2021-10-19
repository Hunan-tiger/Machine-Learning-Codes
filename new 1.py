        self.n_samples, self.n_features = X.shape
        # ���������������ϸ���
        Pypa = {}
        # �������ϸ��ʵĵ���������
        Pxypa = {}
        yset = np.unique(y)
        # ��һ���ǲ�ͬ�ķ���
        for yi in yset:
            Pypa[yi] = {}; Pxypa[yi] = {}
            
            # �ڶ����ǲ�ͬ�ĳ������ԣ����������ֵ�򣬲��ܵ�����������ɢֵ������������ 
            for paIdx in range(self.n_features):
                if columnsMark[paIdx] == 1:
                    continue
                Pypa[yi][paIdx] = {}; Pxypa[yi][paIdx] = {}
                paset = np.unique(X[:, paIdx])
                
                # �������ǲ�ͬ�ĳ������Ե�����ֵ�����������Ӧ��Xarr����yarr
                for pai in paset:
                    yi_pai_idx = np.nonzero((X[:,paIdx]==pai)&(y==yi))#���ز���0��Ԫ�ص�����ֵ
                    
#                    if paIdx==2 and pai==1:
#                        print(yi, '\n', yi_pai_idx)
                    
                    yarr = y[yi_pai_idx]
                    ## ���������������ϸ���
                    Pypa[yi][paIdx][pai] = self.__calyproba(yarr, self.n_samples, len(yset), len(paset))
                    Pxypa[yi][paIdx][pai] = {}
                    
                    # ���Ĳ��ǲ�ͬ���������������ǳ�����������������ɢ����ɢͳ�ƣ�����������ͳ��
                    for xiIdx in range(self.n_features):
                        if xiIdx == paIdx:
                            continue
                        allxiset = np.unique(X[:, xiIdx])
                        Xarr = X[yi_pai_idx, xiIdx].flatten()#Ĭ�ϰ��н�ά��һά��ֻ��������
                        if columnsMark[xiIdx] == 0:
                            ## ������ɢ��������������
                            Pxypa[yi][paIdx][pai][xiIdx] = self.__categorytrain(Xarr, allxiset)
                        else:
                            ## ����������������������
                            Pxypa[yi][paIdx][pai][xiIdx] = self.__continuoustrain(Xarr)
                        
#                        if xiIdx == 4 and paIdx==2 and pai==1:
#                            print(Xarr)
                        
        print('P(y,pa)ѵ�����!')
        print('P(x|y,pa)ѵ�����!')
        self.yProba = Pypa
        self.xyProba = Pxypa
        self.trainSet = X
        self.trainLabel = y
        self.columnsMark = columnsMark        
        return
    
    
    # ������ɢ��������������
    def __categorytrain(self, Xarr, xiset):
        pxypa = {}
        for xivalue in xiset:
            pxypa[xivalue] = {}
            pxypa[xivalue]['count'] = sum(Xarr==xivalue) + self.ls
            pxypa[xivalue]['ratio'] = self.classifyProba(xivalue, Xarr, len(xiset))
        return pxypa
    
    # �������������ľ�ֵ�ͱ�׼��
    def __continuoustrain(self, Xarr):
        pxypa = (Xarr.mean(), Xarr.std())
        return pxypa
        
    # �����������ϸ���
    def __calyproba(self, yarr, ysum, ysetsum, pasetsum):
        yproba = {}
        yproba['count'] = len(yarr) + self.ls
        yproba['ratio'] = (len(yarr) + self.ls) / (ysum + ysetsum * pasetsum)
        return yproba
    
    
    # Ԥ��
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
            ['����', '����', '����', '����', '����', 'Ӳ��', '�ù�'],
            ['�ں�', '����', '����', '����', '����', 'Ӳ��', '�ù�'],
            ['�ں�', '����', '����', '����', '����', 'Ӳ��', '�ù�'],
            ['����', '����', '����', '����', '����', 'Ӳ��', '�ù�'],
            ['ǳ��', '����', '����', '����', '����', 'Ӳ��', '�ù�'],
            ['����', '����', '����', '����', '�԰�', '��ճ', '�ù�'],
            ['�ں�', '����', '����', '�Ժ�', '�԰�', '��ճ', '�ù�'],
            ['�ں�', '����', '����', '����', '�԰�', 'Ӳ��', '�ù�'],
            ['�ں�', '����', '����', '�Ժ�', '�԰�', 'Ӳ��', '����'],
            ['����', 'Ӳͦ', '���', '����', 'ƽ̹', '��ճ', '����'],
            ['ǳ��', 'Ӳͦ', '���', 'ģ��', 'ƽ̹', 'Ӳ��', '����'],
            ['ǳ��', '����', '����', 'ģ��', 'ƽ̹', '��ճ', '����'],
            ['����', '����', '����', '�Ժ�', '����', 'Ӳ��', '����'],
            ['ǳ��', '����', '����', '�Ժ�', '����', 'Ӳ��', '����'],
            ['�ں�', '����', '����', '����', '�԰�', '��ճ', '����'],
            ['ǳ��', '����', '����', 'ģ��', 'ƽ̹', 'Ӳ��', '����'],
            ['����', '����', '����', '�Ժ�', '�԰�', 'Ӳ��', '����']
        ]
    #����ֵ�б�
    labels = ['ɫ��', '����', '�û�', '����', '�겿', '����']