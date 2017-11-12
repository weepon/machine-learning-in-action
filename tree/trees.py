'''
Created on Nov 10, 2017
Decision Tree Source Code for Machine Learning in Action Ch. 3
author: weepon
blog: http://blog.csdn.net/u013829973
Modify:
    2017-11-10
'''
from math import log
import operator
import matplotlib.pyplot as plt
import pickle
'''
函数说明:创建数据集
Parameters:
    无
Returns:
    dataSet - 数据集
    labels - 分类属性(特征)
'''
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],                        #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['age', 'job', 'house', 'credit'] #分类属性
    #转化成离散值
    return dataSet, labels

'''
函数说明:计算数据集香农熵
Parameters:
    dataSet - 数据集
Returns:
    shannonEnt 香农熵
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)                           # 计算数据集的实例总数
    labelCounts = {}
    for featVec in dataSet:                             # 计算每个类别出现的次数的字典
        currentLabel = featVec[-1]                      # 取每个样本的类别，并计算出现的次数
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:                            #计算香农熵
        prob = float(labelCounts[key])/numEntries      # 计算该类别的概率
        shannonEnt -= prob * log(prob,2)               # 利用公式，计算熵
    return shannonEnt

'''
函数说明:按照给定特征划分数据集

Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征索引
    value - 需要返回的特征的值
Returns:
    retDataSet 划分后的数据集
'''    
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:                    ##遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #删掉axis的特征，保留剩下的特征并存到retDataSet
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

'''
函数说明:选择最优特征

Parameters:
    dataSet - 数据集
Returns:
    bestFeature - 信息增益最大的(最优)特征的索引值
'''
    
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #特征数
    baseEntropy = calcShannonEnt(dataSet)  #计算香农熵
    bestInfoGain = 0.0;                 #初始化最大信息增益变量
    bestFeature = -1
    for i in range(numFeatures):        #遍历所有特征
        featList = [example[i] for example in dataSet]#取所有样本的第一个特征
        uniqueVals = set(featList)       #去重复值
        newEntropy = 0.0
        for value in uniqueVals:        #按照第i个特征划分数据下的香农熵，信息增益
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     # 信息增益
        #print("第%d个特征的增益为%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):       #选择最大增益的特征索引
            bestInfoGain = infoGain         
            bestFeature = i
    return bestFeature                      

'''
函数说明:统计classList中出现此处最多的元素(投票)

Parameters:
    classList - 类标签列表
Returns:
    sortedClassCount[0][0] - 出现此处最多的元素(类标签)
'''
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

'''
函数说明:创建决策树（递归函数）

Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签
Returns:
    myTree - 决策树
'''
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]  #取数据集的类别标签
    if classList.count(classList[0]) == len(classList): 
        return classList[0]   #递归停止条件一：如果类别完全相同则停止继续划分
    if len(dataSet[0]) == 1:   #递归停止条件二：遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) #选择最优特征
    bestFeatLabel = labels[bestFeat]             # #最优特征的标签
    myTree = {bestFeatLabel:{}}             #根据最优特征的标签生成树
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #复制标签，递归创建决策树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            

'''
函数说明:获取决策树叶子结点的数目

Parameters:
    myTree - 决策树
Returns:
    numLeafs - 决策树的叶子结点的数目
'''
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

'''
函数说明:获取决策树的层数

Parameters:
    myTree - 决策树
Returns:
    maxDepth - 决策树的层数
'''
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

'''
函数说明:绘制结点

Parameters:
    nodeTxt - 结点名
    centerPt - 文本位置
    parentPt - 标注的箭头位置
    nodeType - 结点格式
Returns:
    无
'''
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")                                            #定义箭头格式
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
    
'''
函数说明:标注有向边属性值

Parameters:
    cntrPt、parentPt - 用于计算标注位置
    txtString - 标注的内容
Returns:
    无
'''    
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

    '''
函数说明:绘制决策树

Parameters:
    myTree - 决策树(字典)
    parentPt - 标注的内容
    nodeTxt - 结点名
Returns:
    无
'''
def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")                                        #设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")                                            #设置叶结点格式    
    numLeafs = getNumLeafs(myTree) 
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]  
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)  #中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD             #y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#测试该结点是否为字典，如果不是字典，代表此结点为叶子结点  
            plotTree(secondDict[key],cntrPt,str(key))       
        else:  
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW  #不是叶结点，递归调用继续绘制
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))#如果是叶结点，绘制叶结点，并标注有向边属性值
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
'''
函数说明:创建绘制面板

Parameters:
    inTree - 决策树(字典)
Returns:
    无
'''
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()                                                #清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops) #去掉x、y轴
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()    

'''
函数说明:使用决策树分类

Parameters:
    inputTree - 已经生成的决策树
    featLabels - 存储选择的最优特征标签
    testVec - 测试数据列表，顺序对应最优特征标签
Returns:
    classLabel - 分类结果
'''    
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]                                                      #获取决策树结点
    secondDict = inputTree[firstStr]                                                        #下一个字典
    featIndex = featLabels.index(firstStr)                                               
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel

'''
函数说明:存储决策树

Parameters:
    inputTree - 已经生成的决策树
    filename - 决策树的存储文件名
Returns:
    无
'''
def storeTree(inputTree, filename):
    
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)
'''
函数说明:读取决策树

Parameters:
    filename - 决策树的存储文件名
Returns:
    pickle.load(fr) - 决策树字典
'''
def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)    
    
if __name__ == '__main__':
    dataSet, labels = createDataSet()
    print(dataSet)
    print('香农熵为：',calcShannonEnt(dataSet))
    print('开始计算最优特征：')
    print("最优特征索引值:" + str(chooseBestFeatureToSplit(dataSet)))
    myTree = createTree(dataSet, labels)
    print('创建好的决策树：',myTree)
    createPlot(myTree)
    testVec = [0,1]       
    featLabels =['house','job']  
    result = classify(myTree, featLabels, testVec)
    print(result)
    storeTree(myTree, 'classifierStorage.txt')
    myTree1 = grabTree('classifierStorage.txt')
    print(myTree1)
    # 下面更换为隐形眼镜数据集
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    print(lenses)
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    myTree_lenses = createTree(lenses, lensesLabels)
    createPlot(myTree_lenses)