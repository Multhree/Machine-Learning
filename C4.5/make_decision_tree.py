#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/1/28 下午 08:51
# @Author  : YuXin Chen

from math import log
import operator
import treePlotter

def createDataSet():
    '''
    :return: 自定义的数据集以及标签
    :function: 生成自定义的数据集以及标签用于训练
    '''
    dataSet = [[0, 0, 0, 0, 'N'],
               [0, 0, 0, 1, 'N'],
               [1, 0, 0, 0, 'Y'],
               [2, 1, 0, 0, 'Y'],
               [2, 2, 1, 0, 'Y'],
               [2, 2, 1, 1, 'N'],
               [1, 2, 1, 1, 'Y']]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    return dataSet, labels

def createTestSet():
    '''
    :return: 测试集
    :function: 生成测试集
    '''
    testSet = [[0, 1, 0, 0],
               [0, 2, 1, 0],
               [2, 1, 1, 0],
               [0, 1, 1, 1],
               [1, 1, 0, 1],
               [1, 0, 1, 0],
               [2, 1, 0, 1]]
    return testSet

def calInfo(dataSet):
    '''
    :param dataSet: 数据集
    :return: 信息熵
    :function: 计算数据集信息熵
    '''
    totalEntries = len(dataSet)
    labelNum = {}
    for vec in dataSet:
        label = vec[-1]
        if label not in labelNum.keys():
            labelNum[label] = 1
        else:
            labelNum[label] += 1
        Info = 0.0
        for label in labelNum:
            coe = float(labelNum[label])/totalEntries
            Info -= coe * log(coe,2)
    return Info

def splitDataSet(dataSet,attribute,value):
    '''
    :param dataSet: 父数据集
    :param attribute: 选定的划分属性
    :param value: 划分属性的值
    :return: 子数据集
    :function: 根据划分属性对数据集进行划分
    '''
    subDataSet = []
    for vec in dataSet:
        if vec[attribute] == value:
            subVec = vec[:attribute]
            subVec.extend(vec[attribute+1:])
            subDataSet.append(subVec)
    return subDataSet

def chooseBestAttribute(dataSet):
    '''
    :param dataSet: 数据集
    :return: 最好的划分属性
    :function: 选出信息增益率最大的属性
    '''
    totalAttribute = len(dataSet[0])-1
    maxInfoGainRatio = 0.0
    bestAttribute = -1
    InfoD = calInfo(dataSet)
    for i in range(totalAttribute):
        vecList = []
        for vec in dataSet:
            vecList.append(vec[i])
        allPossibleValues = set(vecList)
        InfoC = 0.0
        inInfo = 0.0
        for value in allPossibleValues:
            subDataSet = splitDataSet(dataSet,i,value)
            coe = float(len(subDataSet)/len(dataSet))
            InfoC += coe * calInfo(subDataSet)
            inInfo += -coe * log(coe,2)
        infoGain = InfoD - InfoC
        if(inInfo == 0):
            continue
        else:
            infoGainRatio = infoGain / inInfo
        if(infoGainRatio > maxInfoGainRatio):
            maxInfoGainRatio = infoGainRatio
            bestAttribute = i
    return bestAttribute

def calMostClass(classList):
    '''
    :param classList: 类的列表
    :return: 多数类
    :function: 数据集已经处理了所有属性，但是类标签依然不是唯一的，采用多数判决的方法决定该子节点的分类
    '''
    classNum = {}
    for kind in classList:
        if kind not in classNum.keys():
            classNum[kind] = 1
        else:
            classNum[kind] += 1
    sortedClassNum = sorted(classNum.iteritems(),key=operator.itemgetter(1), reverse=True)# dict排序之后会变成list
    return sortedClassNum[0][0]

def createDecisionTree(dataSet,labels):
    '''
    :param dataSet: 数据集
    :param labels: 标签
    :return: 决策树
    :function: 构建决策树
    '''
    classList = [vec[-1] for vec in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return calMostClass(classList)
    bestAttribute = chooseBestAttribute(dataSet)
    bestAttributeLabel = labels[bestAttribute]
    decisionTree = {bestAttributeLabel:{}}
    del(labels[bestAttribute])
    bestAttributeValues = []
    for vec in dataSet:
        bestAttributeValues.append(vec[bestAttribute])
    allPossibleValues = set(bestAttributeValues)
    for value in allPossibleValues:
        subLabels = labels[:]
        decisionTree[bestAttributeLabel][value] = createDecisionTree(splitDataSet(dataSet,bestAttribute,value),subLabels)
    return decisionTree

def saveDecisionTree(decisionTree,filename):
    '''
    :param decisionTree: 已生成的决策树
    :param filename: 文件路径
    :return:
    :function: 保存决策树
    '''
    import pickle
    fw = open(filename,'wb')
    pickle.dump(decisionTree,fw)
    fw.close()

def readDecisionTree(filename):
    '''
    :param filename: 文件路径
    :return: 决策树
    :function: 读取决策树
    '''
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

def classify(decisionTree,labels,testVec):
    '''
    :param decisionTree: 决策树
    :param labels: 标签
    :param testVec: 待分类的向量
    :return: 分类结果
    :function: 对某个测试集中的向量进行分类
    '''
    rootLabel = list(decisionTree.keys())[0]
    rootLabelTree = decisionTree[rootLabel]
    rootLabelIndex = labels.index(rootLabel)
    for key in rootLabelTree.keys():
        if testVec[rootLabelIndex] == key:
            if type(rootLabelTree[key]).__name__ == 'dict':
                classLabel = classify(rootLabelTree[key],labels,testVec)
            else:
                classLabel = rootLabelTree[key]
    return classLabel

def classifyAll(decisionTree,labels,testDataSet):
    '''
    :param decisionTree: 决策树
    :param labels: 标签
    :param testDataSet: 待分类的数据集
    :return: 分类结果
    :function: 对某个测试集中进行分类
    '''
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(decisionTree,labels,testVec))
    return classLabelAll

def main():
    dataSet, labels = createDataSet()
    labels_tmp = labels[:] # 拷贝，createDecisionTree会改变labels
    desicionTree = createDecisionTree(dataSet, labels_tmp)
    #saveDecisionTree(desicionTree, 'classifierStorage.txt')
    #desicionTree = grabTree('classifierStorage.txt')
    print('desicionTree:\n', desicionTree)
    treePlotter.createPlot(desicionTree)
    testSet = createTestSet()
    print('classifyResult:\n', classifyAll(desicionTree, labels, testSet))

if __name__ == '__main__':
    main()






