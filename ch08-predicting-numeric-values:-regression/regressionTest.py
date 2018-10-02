# 测试导入数据
import regression
from numpy import *
xArr, yArr = regression.loadDataSet('ex0.txt')
xArr[0:2]

# 测试标准回归
ws = regression.standRegres(xArr, yArr)
ws
xMat = mat(xArr)
yMat = mat(yArr)
yHat = xMat * ws

# 绘制数据集散点图和最佳拟合直线图
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
# 先将数据点按升序排列
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:, 1], yHat)
plt.show()

# 计算预测值和真实值的相关性
yHat = xMat * ws
corrcoef(yHat.T, yMat)

# 测试LWLR
from importlib import reload
reload(regression)
xArr, yArr = regression.loadDataSet('ex0.txt')
yArr[0]
regression.lwlr(xArr[0], xArr, yArr, 1.0)
regression.lwlr(xArr[0], xArr, yArr, 0.001)
yHat = regression.lwlrTest(xArr, xArr, yArr, 1.0)
# yHat = regression.lwlrTest(xArr, xArr, yArr, 0.01)
# yHat = regression.lwlrTest(xArr, xArr, yArr, 0.003)
# 查看yHat的拟合效果
xMat = mat(xArr)
srtInd = xMat[:, 1].argsort(0)
xSort = xMat[srtInd][:, 0, :]
# 绘图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:, 1], yHat[srtInd])
ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
plt.show()

# 示例：鲍鱼年龄
import regression
from importlib import reload
from numpy import *
reload(regression)
abX, abY = regression.loadDataSet('abalone.txt')
yHat01 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
yHat1 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
yHat10 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
# 分析预测误差
regression.rssError(abY[0:99], yHat01.T)
regression.rssError(abY[0:99], yHat1.T)
regression.rssError(abY[0:99], yHat10.T)
# 使用最小的核在新数据上的表现
yHat01 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
regression.rssError(abY[100:199], yHat01.T)
yHat1 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
regression.rssError(abY[100:199], yHat1.T)
yHat10 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
regression.rssError(abY[100:199], yHat10.T)
# 与简单线性回归比较
ws = regression.standRegres(abX[0:99], abY[0:99])
yHat = mat(abX[100:199]) * ws
regression.rssError(abY[100:199], yHat.T.A)

# 测试岭回归
from importlib import reload
reload(regression)
abX, abY = regression.loadDataSet('abalone.txt')
ridgeWeights = regression.ridgeTest(abX, abY)
# 绘图
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()

# 测试前向逐步线性回归
reload(regression)
xArr, yArr = regression.loadDataSet('abalone.txt')
regression.stageWise(xArr, yArr, 0.01, 200)
regression.stageWise(xArr, yArr, 0.001, 5000)
# 与最小二乘法进行比较
xMat = mat(xArr)
yMat = mat(yArr).T
xMat = regression.regularize(xMat)
yM = mean(yMat, 0)
yMat = yMat - yM
weights = regression.standRegres(xMat, yMat.T)
weights.T

# 测试获取数据
from importlib import reload
reload(regression)
lgX = []; lgY = []
regression.setDataCollect(lgX, lgY)
# 训练算法：建立模型
shape(lgX)
lgX1 = mat(ones((58, 5)))
lgX1[:, 1:5] = mat(lgX)
lgX[0]
lgX[1]
ws = regression.standRegres(lgX1, lgY)
ws
lgX1[0] * ws
lgX1[-1] * ws
lgX1[43] * ws

# 交叉验证
regression.crossValidation(lgX, lgY, 10)
regression.ridgeTest(lgX, lgY)