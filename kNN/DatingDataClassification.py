import matplotlib
import matplotlib.pyplot as plt
import kNN as kNN
import numpy as np

'''
Dating data set classification 

@author: Wentao Kuang
'''

#ploting the scattered dataSet
def plotScatter():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    group, lables = kNN.file2matrix('/Users/kuangwentao/PycharmProjects/ML_Algorithm/kNN/DataSets/datingTestSet.txt', 3)
    ax.scatter(group[:, 1], group[:, 2])
    plt.show()

'''
Error rate testing
employed kNN algoritm with k=3
'''
def errorTest():
    testRatio = 0.10 #test data split rate
    group, lables = kNN.file2matrix('/Users/kuangwentao/PycharmProjects/ML_Algorithm/kNN/DataSets/datingTestSet.txt', 3)
    normgroup, ranges, minVal = kNN.norm(group)
    m = normgroup.shape[0] #number of observations
    numTestVecs = int(m*testRatio) #number of test data
    errorCount = 0.0
    for i in range(numTestVecs):
        result = kNN.kNNClisifier(normgroup[i, :], normgroup[numTestVecs:m, :], lables[numTestVecs:m], 3)
        print "classifier result: %s, the real value is: %s" % (result, lables[i])
        if(result != lables[i]) : errorCount += 1.0
    print "the totoal error rate is %f" % (errorCount/float(numTestVecs))

'''
Clissify with input observation
employed kNN algoritm with k=3
'''
def kNNclassifier():
    ob1 = float(raw_input("Please input the first feature value?"))
    ob2 = float(raw_input("Please input the second feature value?"))
    ob3 = float(raw_input("Please input the third feature value?"))
    group, lables = kNN.file2matrix('/Users/kuangwentao/PycharmProjects/ML_Algorithm/kNN/DataSets/datingTestSet.txt', 3)
    normgroup, ranges, minVals = kNN.norm(group)
    obArr = np.array([ob1, ob2, ob3])
    result = kNN.kNNClisifier((obArr-minVals)/ranges, normgroup, lables, 3)
    #print(result)
    return result




