from math import log

'''
Decision Tree (DT)

@author: Wentao Kuang
'''


'''
calculate Shannon Entropy = -sum_{1,n} p(x_{i})*log_{2}*p_{i}

Input:      dataSet: Matrix data set
        
Output:     Shannon Entropy
'''
def ShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt