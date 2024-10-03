import pandas as pd
import helper.helper as hp

class MeanCentered :
    def __init__(self,data,opsional):
        self.data = data if opsional != 0 else hp.reverseMatrix(data)
        self.opsional = opsional
        self.meanList = [self.mean(i) for i in self.data]
        self.mean_centered_result = self.mean_centered_measure()

    @staticmethod
    def mean(data) :
        return  sum([j for j in data if j != 0]) / len([j for j in data if j != 0])

    def mean_centered_measure(self) :

        if self.opsional == 1 :
            return  [[
                    (self.data[i][j]-self.meanList[i] if self.data[i][j] != 0 else 0) for j in range(len(self.data[i]))
                ] for i in range(len(self.data))
            ]
        else :
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
            return result
    
    def getMeanCenteredArray(self) :
        return self.mean_centered_result
    
    def getMeanCenteredDataFrame(self) :
        return pd.DataFrame(self.mean_centered_result)
    
    def getMeanListArray(self) :
        return self.meanList
    
    def getMeanListDataFrame(self) :
        return pd.DataFrame(self.meanList)
    