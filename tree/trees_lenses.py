'''
Created on Nov 10, 2017
Decision Tree Source Code for Machine Learning in Action Ch. 3
author: weepon
blog: http://blog.csdn.net/u013829973
Modify:
    2017-11-10
'''
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import pandas as pd

lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate','class']#特征标签+类别标签
feature = ['age', 'prescript', 'astigmatic', 'tearRate']#特征标签
lenses = pd.read_table('lenses.txt',names=lensesLabels, sep='\t')
#names：设置列名 ，sep:分隔的正则表达式,'/t'表示以tab键进行分割
lenses_feature = lenses[feature]                        # 特征数据
le = LabelEncoder()                                     #创建LabelEncoder()对象，用于序列化           
for col in lenses_feature.columns:                             #分列序列化
    lenses_feature[col] = le.fit_transform(lenses_feature[col])
clf = tree.DecisionTreeClassifier(max_depth = 4)               #创建DecisionTreeClassifier()类
model = clf.fit(lenses_feature.values, lenses['class'])       #使用数据，构建决策树
print(model)               # 查看当前模型参数

#预测
pre = model.predict([[0,1,0,1]])
print('预测结果为',pre)