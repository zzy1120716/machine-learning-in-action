import logRegres
dataArr, labelMat = logRegres.loadDataSet()
logRegres.gradAscent(dataArr, labelMat)

# 测试梯度上升
from importlib import reload
reload(logRegres)
weights = logRegres.gradAscent(dataArr, labelMat)
logRegres.plotBestFit(weights.getA())

# 测试随机梯度上升
from numpy import *
from importlib import reload
reload(logRegres)
dataArr, labelMat = logRegres.loadDataSet()
weights = logRegres.stocGradAscent0(array(dataArr), labelMat)
logRegres.plotBestFit(weights)

# 测试随机梯度上升改进算法
from numpy import *
from importlib import reload
reload(logRegres)
dataArr, labelMat = logRegres.loadDataSet()
weights = logRegres.stocGradAscent1(array(dataArr), labelMat)
logRegres.plotBestFit(weights)

weights = logRegres.stocGradAscent1(array(dataArr), labelMat, 500)
logRegres.plotBestFit(weights)

# 测试病马死亡预测
from importlib import reload
reload(logRegres)
logRegres.multiTest()