from numpy import *
import regTrees
testMat = mat(eye(4))
testMat
# 按指定列的某个值切分该矩阵
mat0, mat1 = regTrees.binSplitDataSet(testMat, 1, 0.5)
mat0
mat1

# 从测试数据生成一棵回归树
from importlib import reload
reload(regTrees)
from numpy import *
myDat = regTrees.loadDataSet('ex00.txt')
myMat = mat(myDat)
regTrees.createTree(myMat)

# 多次切分
myDat1 = regTrees.loadDataSet('ex0.txt')
myMat1 = mat(myDat1)
regTrees.createTree(myMat1)

# 预剪枝
regTrees.createTree(myMat, ops=(0,1))
myDat2 = regTrees.loadDataSet('ex2.txt')
myMat2 = mat(myDat2)
regTrees.createTree(myMat2)
regTrees.createTree(myMat2, ops=(10000, 4))

# 后剪枝
from importlib import reload
reload(regTrees)
myTree = regTrees.createTree(myMat2, ops=(0,1))
myDatTest = regTrees.loadDataSet('ex2test.txt')
myMat2Test = mat(myDatTest)
# 剪枝
regTrees.prune(myTree, myMat2Test)

# 模型树
from importlib import reload
reload(regTrees)
myMat2 = mat(regTrees.loadDataSet('exp2.txt'))
regTrees.createTree(myMat2, regTrees.modelLeaf, regTrees.modelErr, (1, 10))

# 示例：自行车与人智商的关系
from importlib import reload
reload(regTrees)
trainMat = mat(regTrees.loadDataSet('bikeSpeedVsIq_train.txt'))
testMat = mat(regTrees.loadDataSet('bikeSpeedVsIq_test.txt'))
myTree = regTrees.createTree(trainMat, ops=(1,20))
yHat = regTrees.createForeCast(myTree,testMat[:,0])
corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
# 再创建一棵模型树
myTree2 = regTrees.createTree(trainMat, regTrees.modelLeaf, regTrees.modelErr,ops=(1,20))
yHat2 = regTrees.createForeCast(myTree2, testMat[:,0], regTrees.modelTreeEval)
corrcoef(yHat2, testMat[:,1], rowvar=0)[0, 1]
# 查看标准的线性回归效果
ws, X, Y = regTrees.linearSolve(trainMat)
ws
# 得到测试集上所有的yHat预测值
for i in range(shape(testMat)[0]):
    yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]

corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]

# 用Tkinter创建GUI
from tkinter import *
root = Tk()
myLabel = Label(root, text="Hello World")
myLabel.grid()
root.mainloop()