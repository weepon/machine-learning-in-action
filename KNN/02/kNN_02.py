'''
Created on Sep 10, 2017

kNN: k近邻（k Nearest Neighbors）
实战：使用k近邻算法改进约会网站的配对效果

author：we-lee
'''
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
'''
k近邻算法

Input:      inX: 测试集 (1xN)
            dataSet: 已知数据的特征(NxM)
            labels: 已知数据的标签或类别(1xM vector)
            k: k近邻算法中的k            
Output:     测试样本最可能所属的标签

'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] # shape[0]返回dataSet的行数
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet # tile(inX,(a,b))函数将inX重复a行，重复b列
    sqDiffMat = diffMat**2 #作差后平方
    sqDistances = sqDiffMat.sum(axis=1)#sum()求和函数，sum(0)每列所有元素相加，sum(1)每行所有元素相加
    distances = sqDistances**0.5  #开平方，求欧式距离
    sortedDistIndicies = distances.argsort() #argsort函数返回的是数组值从小到大的索引值  
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #取出前k个距离对应的标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1#计算每个类别的样本数。字典get()函数返回指定键的值，如果值不在字典中返回默认值0
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #reverse降序排列字典
    #python2版本中的iteritems()换成python3的items()
    #key=operator.itemgetter(1)按照字典的值(value)进行排序
    #key=operator.itemgetter(0)按照字典的键(key)进行排序
    return sortedClassCount[0][0] #返回字典的第一条的key，也即是测试样本所属类别
   
'''
将文本程序转化为numpy矩阵

Input:      filename：     文件名          
Output:     returnMat：    特征矩阵
            classLabelVector：标签向量
'''
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)            #得到文件行数
    returnMat = np.zeros((numberOfLines,3))     #创建返回的numpy矩阵
    classLabelVector = []                       #类别标签初始化   
    index = 0
    for line in arrayOLines:
        line = line.strip()                   #截取掉所有的回车字符
        listFromLine = line.split('\t')       #使用tab字符\t将上一行得到的整行数据分割成一个元素列表
        returnMat[index,:] = listFromLine[0:3]#截取前三个元素，存储到特征矩阵中
        if listFromLine[-1] == 'largeDoses':  #极具魅力的人记为1
            classLabelVector.append(1)
        if listFromLine[-1] == 'smallDoses': #极具魅力的人记为2
            classLabelVector.append(2)
        if listFromLine[-1] == 'didntLike': #极具魅力的人记为3
            classLabelVector.append(3)
        index += 1
    return returnMat,classLabelVector

    
'''
可视化程序

'''
def datavisualization(datingDataMat,datingLabels):
    zhfont = matplotlib.font_manager.FontProperties(fname=r'c:\windows\fonts\simsun.ttc')#设置中文字体路径
    fig = plt.figure(figsize=(13,8))#新建画布，并定义大小
    ax1 = fig.add_subplot(221)#画布切分成2x2，第一个位置添加子图
    ax1.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
    ax1.axis([-2,25,-0.2,2.0])#利用datingLabel变量，绘制色彩不等，尺寸不同的点;指定坐标轴范围
    plt.xlabel('玩视频游戏所消耗时间占百分比',fontproperties=zhfont)#横轴标签
    plt.ylabel('每周消费的冰激凌公升数',fontproperties=zhfont) #纵轴标签
    ax2 = fig.add_subplot(222)#在画布上添加第二个子图
    ax2.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
    ax2.axis([-5000,100000,-2,23])
    plt.xlabel('每年获得的飞行常客里程数',fontproperties=zhfont)#横轴标签
    plt.ylabel('玩视频游戏所消耗时间占百分比',fontproperties=zhfont)#纵轴标签
    ax3 = fig.add_subplot(223)#在画布上添加第三个子图
    ax3.scatter(datingDataMat[:,0], datingDataMat[:,2], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
    ax3.axis([-5000,100000,-0.2,2.0])
    plt.xlabel('每年获得的飞行常客里程数',fontproperties=zhfont)#横轴标签
    plt.ylabel('每周消费的冰激凌公升数',fontproperties=zhfont)#纵轴标签
    plt.show()
    
    
'''
函数功能：归一化特征值
Input:      dataSet：     特征矩阵          
Output:     normDataSet： 归一化后的特征矩阵
            ranges：      取值范围（最大值与最小值之差）
            minVals：     最小值
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)#从列中选取最小值
    maxVals = dataSet.max(0)#从列中选取最大值
    ranges = maxVals - minVals#计算可能的取值范围
    normDataSet = np.zeros(np.shape(dataSet))#初始化矩阵，维数(样本数x特征数)
    m = dataSet.shape[0]#获取dataSet的行数
    normDataSet = dataSet - np.tile(minVals, (m,1)) #归一化公式的分子
    normDataSet = normDataSet/np.tile(ranges, (m,1))   #归一化公式
    return normDataSet, ranges, minVals

'''
分类器对约会网站的测试函数

''' 
def datingClassTest():
    hoRatio = 0.10                     #将数据集且分为训练集和测试集的比例
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt') #加载原始数据集
    normMat, ranges, minVals = autoNorm(datingDataMat)#归一化
    m = normMat.shape[0]              #取矩阵行数
    numTestVecs = int(m*hoRatio)        #取整
    errorCount = 0.0                 #分错的个数
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ('the classifier came back with: %d, the real answer is: %d' % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0     #计算分错的样本数
    print ('the total error rate is: %f' % (errorCount/float(numTestVecs)))
    print ('分错的个数:',errorCount)

'''
函数功能:通过输入一个人的三维特征,进行分类输出
Input:                
Output:
'''
def classifyPerson():
    #输出结果
    resultList = ['讨厌','有些喜欢','非常喜欢']
    #三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    #打开的文件名
    filename = "datingTestSet.txt"
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    #训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #生成NumPy数组,测试集
    inArr = np.array([precentTats, ffMiles, iceCream])
    #测试集归一化
    norminArr = (inArr - minVals) / ranges
    #返回分类结果
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    #打印结果
    print("你可能%s这个人" % (resultList[classifierResult-1]))
    
'''
主程序
'''   
if __name__ =='__main__':
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    print('datingDataMat:\n',datingDataMat)
    print('datingLabels:\n',datingLabels)
    datavisualization(datingDataMat,datingLabels)
    datingClassTest() 
    classifyPerson()    
