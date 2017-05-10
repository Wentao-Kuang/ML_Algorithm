import matplotlib
import matplotlib.pyplot as plt
import kNN as kNN

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

plotScatter()