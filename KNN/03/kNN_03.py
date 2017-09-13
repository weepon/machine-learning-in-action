'''
Created on Sep 10, 2017

kNN: k近邻（k Nearest Neighbors）
实战：手写识别系统

author：weepon
'''
import numpy as np
import operator
from os import listdir
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
函数功能：将32x32的二进制图像转换为1x1024向量

Input:     filename :文件名
Output:    二进制图像的1x1024向量

'''
def img2vector(filename):
    returnVect = np.zeros((1,1024))            #创建空numpy数组
    fr = open(filename)                         #打开文件
    for i in range(32):
        lineStr = fr.readline()                #读取每一行内容
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])#将每行前32个字符值存储在numpy数组中
    return returnVect

'''
函数功能：手写数字分类测试
'''    
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')    #加载训练集
    m = len(trainingFileList)                     #计算文件夹下文件的个数，因为每一个文件是一个手写体数字
    trainingMat = np.zeros((m,1024))            #初始化训练向量矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]        #获取文件名
        fileStr = fileNameStr.split('.')[0]     #从文件名中解析出分类的数字
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #加载测试集
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]          #从文件名中解析出测试样本的类别
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3) #开始分类
        print ('the classifier came back with: %d, the real answer is: %d' % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0            #计算分错的样本数
    print ('\nthe total number of errors is: %d' % errorCount)
    print ('\nthe total error rate is: %f' % (errorCount/float(mTest)))

'''
主函数
'''    
if __name__ == '__main__':
    handwritingClassTest()
        
    
    
