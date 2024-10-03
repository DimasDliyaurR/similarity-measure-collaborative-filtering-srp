import meanCentered as mc
import numpy as np
import helper.helper as hp

class Prediction() :
    def __init__(self, meanC,similarity,data,*,opsional,k):
        self.similarity = similarity
        self.mean_centered = meanC
        self.data = data
        self.k = k
        self.opsional = opsional
        self.tes = self.selectedNeighborhood(similarity,0,2,2,data,meanC,self.opsional)
    
    @staticmethod
    def numerator(similarity,meanCentered) :
        return sum(sim*meanC for sim,meanC in zip(similarity,meanCentered))
    
    @staticmethod
    def denominator(similarity) :
        return sum(abs(sim) for sim in similarity)
    
    @staticmethod
    def selectedNeighborhood(neighborhood,index,indexUser,k,data,meanCentered,opsional) -> list :
        indexZero = hp.checkIndexZeroOfData(data,indexUser if opsional == 0 else index,len(neighborhood))
        indexOfNeighborhood = list(np.delete(hp.createList(0,len(neighborhood[index])-1),indexZero).tolist())
        neighborhood = list(np.delete(neighborhood[index],indexZero).tolist())

        lengthLoop = len(neighborhood)
        for i in range(lengthLoop-2,-1,-1) :
            indexFlag = i
            prevNeighborhood = np.real(neighborhood[i])
            prevIndexList = indexOfNeighborhood[i]
            innerCondition = True
            j=i+1
            while innerCondition and j < lengthLoop and prevNeighborhood < np.real(neighborhood[j]) :
                if prevNeighborhood < np.real(neighborhood[j]) :
                    indexFlag = j
                    neighborhood[j-1] = np.real(neighborhood[j])
                    indexOfNeighborhood[j-1] = indexOfNeighborhood[j]
                    j+=1
                else :
                    j+=1
                    innerCondition = False
            neighborhood[indexFlag] = prevNeighborhood
            indexOfNeighborhood[indexFlag] = prevIndexList

        indexOfNeighborhood = [
            (meanCentered[i][indexUser]) 
            for i in indexOfNeighborhood[0:k]
            ]

        return [neighborhood[0:k],indexOfNeighborhood]
    
    def prediction_measure(self,userTarget,item) -> float :
        target = self.selectedNeighborhood(self.similarity,item,userTarget,self.k,self.data,self.mean_centered,self.opsional)
