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