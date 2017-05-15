'''
Decision Tree (DT)

@author: Wentao Kuang
'''

import matplotlib.pyplot as plt

'''

'''
decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
leafNode = dict(boxstyple = "round4", fc="0.8")
arraw_args = dict(arrowstyle = "<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1, facecolor= 'white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('DecisionNode', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('leafNode', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

createPlot()