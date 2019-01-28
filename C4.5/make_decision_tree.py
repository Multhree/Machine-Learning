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
    :function: 生成自定义的数据集以及标签
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
def calInfo(dataSet):
    '''
    :param dataSet: 数据集
    :return: 信息熵
    :function: 计算数据集信息熵
    '''
    totalEntries = len(dataSet)
    labelNum = {}
    for vec in dataSet:
        label = dataSet[-1]
        if label not in labelNum.keys():
            labelNum[label] = 1
        else:
            labelNum[label] += 1
        Info = 0.0
        for label in labelNum:
            coe = float(labelNum[label])/totalEntries
            Info -= coe * log(coe,2)
        return Info

def splitDataSet(dataSet,property,value):
    '''
    :param dataSet: 父数据集
    :param property: 选定的划分属性
    :param value: 划分属性的值
    :return: 子数据集
    :function: 根据划分属性对数据集进行划分
    '''
    subDataSet = []
    for vec in dataSet:
        if vec[property] == value:
            subVec = vec[:property]
            subVec.extend(vec[property+1:])
            subDataSet.append(subVec)
    return subDataSet

def chooseBestProperty(dataSet):
    '''
    :param dataSet: 数据集
    :return: 最好的划分属性
    :function: 选出信息增益率最大的属性
    '''
    totalProperty = len(dataSet[0])-1
    maxInfoGainRatio = 0.0
    bestProperty = -1
    InfoD = calInfo(dataSet)
    for i in range(totalProperty):
        vecList = []
        for vec in dataSet:
            vecList.append(vec[i])
        allPossibleValue = set(vecList)
        InfoC = 0.0
        inInfo = 0.0
        for value in allPossibleValue:
            subDataSet = splitDataSet(dataSet,i,value)
            coe = float(len(subDataSet)/len(dataSet))
            InfoC += coe * calInfo(dataSet)
            inInfo -= coe * log(coe,2)
        infoGain = InfoD - InfoC
        if(inInfo == 0):
            continue
        else:
            infoGainRatio = infoGain / inInfo
        if(infoGainRatio > maxInfoGainRatio):
            maxInfoGainRatio = infoGainRatio
            bestProperty = i
    return bestProperty

def calMostClass(classList):
    '''
    :param classList: 类的列表
    :return: 多数类
    :function: 数据集已经处理了所有属性，但是类标签依然不是唯一的，采用多数判决的方法决定该子节点的分类
    '''
    classNum = {}
    for kind in classList:
        if kind in classNum.keys():
            classNum[kind] = 1
        else:
            classNum[kind] += 1
    sortedClassNum = sorted(classNum.iteritems(),key=operator.itemgetter(1), reverse=True)# dict排序之后会变成list
    return sortedClassNum[0][0]


