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

# 测试分类函数
myDat, labels = trees.createDataSet()
labels
myTree = treePlotter.retrieveTree(0)
myTree
trees.classify(myTree, labels, [1, 0])
trees.classify(myTree, labels, [1, 1])

from importlib import reload
reload(trees)
# 测试pickle决策树存储
trees.storeTree(myTree, 'classifierStorage.txt')
trees.grabTree('classifierStorage.txt')

# 加载隐形眼镜数据
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses, lensesLabels)
lensesTree
treePlotter.createPlot(lensesTree)