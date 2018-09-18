import trees

# 测试计算香农熵
from importlib import reload
reload(trees)
myDat, labels = trees.createDataSet()
trees.calcShannonEnt(myDat)

myDat[0][-1] = 'maybe'
trees.calcShannonEnt(myDat)

# 测试函数splitDataSet
reload(trees)
myDat, labels = trees.createDataSet()
trees.splitDataSet(myDat, 0, 1)
trees.splitDataSet(myDat, 0, 0)

# 测试chooseBestFeatureToSplit
reload(trees)
myDat, labels = trees.createDataSet()
trees.chooseBestFeatureToSplit(myDat)