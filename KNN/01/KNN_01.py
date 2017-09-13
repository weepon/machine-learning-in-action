# -*- coding: utf-8 -*-
'''
Created on Sep 10, 2017

kNN: k近邻（k Nearest Neighbors） 电影分类
  
author：we-lee
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator

'''
函数功能：创建数据集
Input:     无
Output:     group：数据集
            labels：类别标签
'''
def createDataSet():#创建数据集
    group = np.array([[3,104],[2,100],[99,5],[98,2]])
    labels = ['爱情片','爱情片','动作片','动作片']
    return group, labels
    
'''
函数功能：   kNN分类
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
        voteIlabel = labels[sortedDistIndicies[i]]#取出前k个距离对应的标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1#计算每个类别的样本数。字典get()函数返回指定键的值，如果值不在字典中返回默认值0
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #reverse降序排列字典
    #python2版本中的iteritems()换成python3的items()
    #key=operator.itemgetter(1)按照字典的值(value)进行排序
    #key=operator.itemgetter(0)按照字典的键(key)进行排序
    return sortedClassCount[0][0] #返回字典的第一条的key，也即是测试样本所属类别


'''
函数功能：   主函数    
'''    
if __name__ == '__main__':
    group,labels = createDataSet()#创建数据集
    print('group:\n',group)#打印数据集
    print('labels:',labels)
    zhfont = matplotlib.font_manager.FontProperties(fname=r'c:\windows\fonts\simsun.ttc')#设置中文字体路径
    fig = plt.figure(figsize=(10,8))#可视化
    ax = plt.subplot(111)          #图片在第一行，第一列的第一个位置
    ax.scatter(group[0:2,0],group[0:2,1],color='red',s=50)
    ax.scatter(group[2:4,0],group[2:4,1],color='blue',s=50)
    ax.scatter(18,90,color='orange',s=50)
    plt.annotate('which class?', xy=(18, 90), xytext=(3, 2),arrowprops=dict(facecolor='black', shrink=0.05),)
    plt.xlabel('打斗镜头',fontproperties=zhfont)
    plt.ylabel('接吻镜头',fontproperties=zhfont)
    plt.title('电影分类可视化',fontproperties=zhfont)
    plt.show()
    testclass = classify0([18,90], group, labels, 3)#用未知的样本来测试算法
    print('测试结果：',testclass)#打印测试结果