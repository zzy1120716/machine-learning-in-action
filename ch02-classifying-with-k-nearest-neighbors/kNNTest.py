import kNN
group, labels = kNN.createDataSet()
kNN.classify0([0, 0], group, labels, 3)

# 读取文件为矩阵
from importlib import reload
reload(kNN)
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')

# 创建散点图
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show()

# 归一化特征值
reload(kNN)
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)

# 执行分类器测试程序
reload(kNN)
kNN.datingClassTest()

# 运行约会小程序
reload(kNN)
kNN.classifyPerson()

# 测试图像文件转换为向量
reload(kNN)
testVector = kNN.img2vector('digits/testDigits/0_13.txt')
testVector[0,0:31]
testVector[0,32:63]

# 测试手写数字识别
reload(kNN)
kNN.handwritingClassTest()