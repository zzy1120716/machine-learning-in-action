import adaboost
datMat, classLabels = adaboost.loadSimpData()

# 了解实际adaboost运行过程
from numpy import *
D = mat(ones((5, 1)) / 5)
adaboost.buildStump(datMat, classLabels, D)

# 测试单层决策树训练
from importlib import reload
reload(adaboost)
classifierArray = adaboost.adaBoostTrainDS(datMat, classLabels, 9)
classifierArray

# 测试分类函数
from importlib import reload
reload(adaboost)
datArr, labelArr = adaboost.loadSimpData()
classifierArr = adaboost.adaBoostTrainDS(datArr, labelArr, 30)
adaboost.adaClassify([0, 0], classifierArr)

# 测试数据加载
from importlib import reload
reload(adaboost)
datArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
classifierArray = adaboost.adaBoostTrainDS(datArr, labelArr, 10)

testArr, testLabelArr = adaboost.loadDataSet('horseColicTest2.txt')
prediction10 = adaboost.adaClassify(testArr, classifierArray)
# 得到错误分类的示例类型数量
errArr = mat(ones((67, 1)))
errArr[prediction10 != mat(testLabelArr).T].sum()

# 测试ROC曲线绘制
from importlib import reload
reload(adaboost)
datArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
classifierArray, aggClassEst = adaboost.adaBoostTrainDS(datArr, labelArr, 10)
adaboost.plotROC(aggClassEst.T, labelArr)