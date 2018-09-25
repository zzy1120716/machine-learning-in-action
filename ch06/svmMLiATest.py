import svmMLiA
dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
labelArr

# 测试smo简化算法
from importlib import reload
reload(svmMLiA)
b, alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
alphas[alphas > 0]
from numpy import *
shape(alphas[alphas > 0])
for i in range(100):
    if alphas[i] > 0.0:
        print(dataArr[i], labelArr[i])

# 测试优化算法
from importlib import reload
reload(svmMLiA)
dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
b, alphas = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40)

# 测试分类
reload(svmMLiA)
ws = svmMLiA.calcWs(alphas, dataArr, labelArr)
ws

datMat = mat(dataArr)
datMat[0] * mat(ws) + b

# 确认分类结果的正确性
labelArr[0]
datMat[2] * mat(ws) + b
labelArr[2]
datMat[1] * mat(ws) + b
labelArr[1]

# 测试带核函数的smo
reload(svmMLiA)
svmMLiA.testRbf()

# 测试手写数字识别
reload(svmMLiA)
svmMLiA.testDigits(('rbf', 20))