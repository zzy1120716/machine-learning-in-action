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

# 测试创建树
reload(trees)
myDat, labels = trees.createDataSet()
myTree = trees.createTree(myDat, labels)
myTree

# 测试matplotlib
import treePlotter
treePlotter.createPlot()

# 测试获取叶子数量及树深度的函数
reload(treePlotter)
treePlotter.retrieveTree(1)
myTree = treePlotter.retrieveTree(0)
treePlotter.getNumLeafs(myTree)
treePlotter.getTreeDepth(myTree)

# 绘制树
reload(treePlotter)
myTree=treePlotter.retrieveTree(0)
treePlotter.createPlot(myTree)

# 变更字典，重新绘制
myTree['no surfacing'][3]='maybe'
myTree
treePlotter.createPlot(myTree)