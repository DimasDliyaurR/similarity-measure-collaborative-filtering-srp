def indexOfZero (data1,data2):
    return [i for i in range(len(data2)) if data1[i] == 0 or data2[i] == 0]

def checkIndexZeroOfData (*,data,index,indexUser):
    result = [i for i in range(len(data[index])) if data[index][i] == 0]
    if not heyStack(indexUser,result) :
        result.append(indexUser)
    return result

def heyStack(needle,haystack) :
    for item in haystack :
        if item == needle :
            return True
    return False

def reverseMatrix(data) :
    return [[float(data[j][i]) for j in range(len(data))]for i in range(len(data[0]))]

def remove_items(test_list, item): 
    # using list comprehension to perform the task 
    res = [i for i in test_list if i != item] 
    return res
 
def isNotExistIndex(index,data): 
    return 0 <= index < len(data)

def createList(r1, r2):
    if (r1 == r2):
        return r1
    else:
        res = []
        while(r1 < r2+1 ):
            res.append(r1)
            r1 += 1
    return res