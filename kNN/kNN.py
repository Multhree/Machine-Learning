#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/7/27 上午 10:37
# @Author  : YuXin Chen

from numpy import *
import operator
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def drawFig(datingDataMat,datingLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()

def file2matrix(filename):
    fr = open(filename,encoding = 'utf-8')
    arrayOfLines = fr.readlines()   #读取文件的每一行
    numberOfLines = len(arrayOfLines) #获得文件行数
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip() #去除首尾空格和回车
        listFromLine = line.split() #按照tab键分割数据
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return  returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)    #参数0可以从选取每一列的最小值组成向量
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # shape[0]表示矩阵有多少行 shape[1]表示矩阵有多少列
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  # 计算Ai-Bi
    sqDiffMat = diffMat**2  #计算(Ai-Bi)^2
    sqDistances = sqDiffMat.sum(axis=1) # 计算(A0-B0)^2+...+(Ai-Bi)^2
    distances = sqDistances**0.5    # 计算((A0-B0)^2+...+(Ai-Bi)^2)^0.5 也就是欧式距离
    sortedDistIndicies = distances.argsort()    # 得到数组的值按递增排序的索引
    classCount = {}
    for i in range (k): #距离最近的k个点
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1    # 如果voteIlabels的key不存在就返回0
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def datingClassTest():
    hoRatio = 0.10  # 10%的数据作为测试集
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))

def classifyPerson():
    resultList = ["not at all", "in small doses", "in large doses"]
    percentTats = float(input("monthly income?"))
    ffMiles = float(input("level of appearance?"))
    iceCream = float(input("running miles per month?"))
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0(inArr, datingDataMat, datingLabels, 3)
    print("You will probably like this person:",resultList[classifierResult-1])

def img2vector(filename):
    returnVector = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0,32*i+j] = int(lineStr[j])
    return returnVector

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir("trainingDigits")
    mTrain = len(trainingFileList)
    trainingMat = zeros((mTrain,1024))
    for i in range(mTrain):
        filenameStr = trainingFileList[i]
        fileStr = filenameStr.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        hwLabels.append(classNum)
        trainingMat[i,:] = img2vector("trainingDigits/%s"%filenameStr)
    testFileList = listdir("testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        filenameStr = testFileList[i]
        fileStr = filenameStr.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        testVector = img2vector("testDigits/%s"%filenameStr)
        classifierResult = classify0(testVector, trainingMat, hwLabels, 4)
        print("the classifier came back with: %d, the real answer is: %d" %(classifierResult, classNum))
        if(classifierResult != classNum):
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))

def main():
    handwritingClassTest()

if __name__ == '__main__':
    main()