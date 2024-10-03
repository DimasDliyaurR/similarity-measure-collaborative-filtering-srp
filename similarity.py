import pandas as pd
import numpy as np
import helper.helper as hp
import cmath
import helper.helper as help
import meanCentered as mc
import prediction as pc

class Pearson(mc.MeanCentered,pc.Prediction) :
    def __init__(self,data,*,opsional,k):
        super().__init__(data,opsional)
        self.result = self.mainSimilarityMeasure()
        pc.Prediction.__init__(self,self.mean_centered_result,self.result,data,opsional=opsional,k=k)

    @staticmethod
    def numerator(data1,data2):
        return sum((data1[i] * data2[i]) for i in range(len(data1)))

    @staticmethod
    def denominator(data1,data2):
        return cmath.sqrt(sum((data1[i]**2) for i in range(len(data1)))) * cmath.sqrt(sum((data2[i]**2) for i in range(len(data1))))

    def measureSimilarity(self,u,v,data,meanC = []):
        tempMc1 = [meanC[u][mc1] for mc1 in range(len(data[u]))]
        tempMc2 = [meanC[v][mc2] for mc2 in range(len(data[v]))]

        tempMc1 = np.delete(tempMc1,help.indexOfZero(data[u],data[v]))
        tempMc2 = np.delete(tempMc2,help.indexOfZero(data[u],data[v]))
        denom = self.denominator(tempMc1,tempMc2)
        return self.numerator(tempMc1,tempMc2) / denom if denom != 0 else 0

    def mainSimilarityMeasure(self) :
        result = [[] for _ in range(len(self.data))]
        
        for i in range(len(self.data)) :
            for j in range(i,len(self.data)):
                if i == j :
                    result[i].append(1)
                    continue
                similarity_result = self.measureSimilarity(i,j,self.data,self.mean_centered_result)
                result[i].append(similarity_result)
                result[j].append(similarity_result)
        return result
    
    def neighborhood(self,u,i):
        return np.sort(i for i,j in zip(self.result[u],self.data[i]) if j != "?")
    
    def getArray(self):
        return np.array(self.result)
        
    def getDataFrame(self) :
        return pd.DataFrame(self.result)


class Cosine(Pearson):
    def __init__(self, data, *, opsional, k):
        super().__init__(data, opsional=opsional, k=k)
    
    def measureSimilarity(self,u,v,data):
        tempMc1 = [data[u][mc1]for mc1 in range(len(data[u]))]
        tempMc2 = [data[v][mc2]for mc2 in range(len(data[v]))]

        tempMc1= np.delete(tempMc1,help.indexOfZero(data[u],data[v]))
        tempMc2 = np.delete(tempMc2,help.indexOfZero(data[u],data[v]))

        denom = self.denominator(data[v],data[u])
        return self.numerator(tempMc1,tempMc2) / denom if denom != 0 else 0

    def mainSimilarityMeasure(self) :
        result = [[] for _ in range(len(self.data))]
        for i in range(len(self.data)) :
            for j in range(i,len(self.data)):
                if i == j :
                    result[i].append(1)
                    continue
                similarity_result = self.measureSimilarity(i,j,self.data)
                result[i].append(similarity_result)
                result[j].append(similarity_result)

        return np.transpose(result)

class ACosine(mc.MeanCentered,pc.Prediction) :

    def __init__(self,data,*,opsional,k):
        super().__init__(data,0 if opsional == 1 else 1)
        self.mean_centered_result = self.mean_centered_measure()
        self.result = self.mainSimilarityMeasure()
        pc.Prediction.__init__(self,self.mean_centered_result,self.result,data,opsional=opsional,k=k)

    @staticmethod
    def numerator(data1,data2):
        return sum((data1[i] * data2[i]) for i in range(len(data1)))

    @staticmethod
    def denominator(data1,data2):
        return cmath.sqrt(sum((data1[i]**2) for i in range(len(data1)))) * cmath.sqrt(sum((data2[i]**2) for i in range(len(data1))))
    
    def mean_centered_measure(self) :
        if self.opsional == 0 :
            return  [[
                    (self.data[i][j]-self.meanList[i] if self.data[i][j] != 0 else 0) for j in range(len(self.data[i]))
                ] for i in range(len(self.data))
            ]
        elif self.opsional == 1 :
            print(len(self.data),len(self.data[0]))
            result = []
            for i in range(len(self.data)) :
                resultInner = []
                for j in range(len(self.data[i])):
                    if self.data[i][j] == 0 :
                        resultInner.append(0)
                        continue
                    m = self.data[i][j] - self.meanList[i]
                    resultInner.append(m)
                result.append(resultInner)
            return hp.reverseMatrix(result)
    
    def measureSimilarity(self,u,v,data,meanC = []):
        tempMc1 = [meanC[u][mc1]for mc1 in range(len(data[u]))]
        tempMc2 = [meanC[v][mc2]for mc2 in range(len(data[v]))]

        tempMc1= np.delete(tempMc1,help.indexOfZero(data[u],data[v]))
        tempMc2 = np.delete(tempMc2,help.indexOfZero(data[u],data[v]))
        denom = self.denominator(tempMc1,tempMc2).real
        return self.numerator(tempMc1,tempMc2) / denom if denom != 0 else 0
    
    def mainSimilarityMeasure(self) :
        result = [[]for _ in range(len(self.data[0]))]
        
        for i in range(len(self.data[0] if self.opsional == 1 else self.data[0])) :
            for j in range(i,len(self.data[0])):
                if i == j :
                    result[i].append(1)
                    continue
                similarity_result = self.measureSimilarity(i,j,hp.reverseMatrix(self.data),self.mean_centered_result if self.opsional == 1 else hp.reverseMatrix(self.mean_centered_result))
                result[i].append(similarity_result)
                result[j].append(similarity_result)
        return result
    
    def getArray(self):
        return np.array(self.result)
        
    def getDataFrame(self) :
        return pd.DataFrame(self.result)


class BC(mc.MeanCentered,pc.Prediction) :
    def __init__(self,data,*,opsional,k):
        super().__init__(data, opsional)
        self.result = self.mainSimilarityMeasure()
        pc.Prediction.__init__(self,self.mean_centered_result,self.result,data,opsional=opsional,k=k)
    
    def probabilitas(self,kolom,target):
        return self.data[kolom].count(target) / (len(self.data[kolom]) - self.data[kolom].count(0))
    
    def bunchOfProbabilitas(self,kolom) :
        highestNumber = int(max([max(number) for number in self.data]))+1
        lowestNumber = int(min([min(hp.remove_items(number,0)) for number in self.data]))
        return [self.probabilitas(kolom,i) for i in range(lowestNumber,highestNumber)]

    def measureSimilarity(self,u,j) :
        target1 = self.bunchOfProbabilitas(u)
        target2 = self.bunchOfProbabilitas(j)
        result = sum([cmath.sqrt(x*y).real for x,y in zip(target1,target2)])
        return result
    
    def mainSimilarityMeasure(self) :
        result = [[] for _ in range(len(self.data))]
        
        for i in range(len(self.data)) :
            for j in range(i,len(self.data)):
                if i == j :
                    result[i].append(1)
                    continue
                similarity_result = self.measureSimilarity(i,j)
                result[i].append(similarity_result)
                result[j].append(similarity_result)
        return result

    def getArray(self):
        return np.array(self.result)
        
    def getDataFrame(self) :
        return pd.DataFrame(self.result)
