'''
Created on Sep 10, 2017

kNN: k近邻（k Nearest Neighbors）
实战：手写识别系统

author：weepon
'''
import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier

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
    #构建kNN分类器
    neigh = KNeighborsClassifier(n_neighbors = 3, algorithm = 'auto')
    #拟合模型, trainingMat为测试矩阵,hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)
    testFileList = listdir('testDigits')        #加载测试集
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]          #从文件名中解析出测试样本的类别
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = neigh.predict(vectorUnderTest) #开始分类
        print ('the classifier came back with: %d, the real answer is: %d' % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0            #计算分错的样本数
    print ('\nthe total number of errors is: %d' % errorCount)
    print ('\nthe total error rate is: %f' % (errorCount/float(mTest)))

'''
主函数
'''    
if __name__ == '__main__':
    handwritingClassTest()
        
    
    
