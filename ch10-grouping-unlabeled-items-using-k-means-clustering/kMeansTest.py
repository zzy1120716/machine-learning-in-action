import kMeans
from numpy import *

# 构建矩阵
datMat = mat(kMeans.loadDataSet('testSet.txt'))
min(datMat[:, 0])
min(datMat[:, 1])
max(datMat[:, 1])
max(datMat[:, 0])
# 支持函数
kMeans.randCent(datMat, 2)
kMeans.distEclud(datMat[0], datMat[1])

# 聚类
from importlib import reload
reload(kMeans)
datMat = mat(kMeans.loadDataSet('testSet.txt'))
myCentroids, clustAssing = kMeans.kMeans(datMat, 4)

# 二分
from importlib import reload
reload(kMeans)
datMat3 = mat(kMeans.loadDataSet('testSet2.txt'))
centList, myNewAssments = kMeans.biKmeans(datMat3, 3)
centList

# 示例
from importlib import reload
reload(kMeans)
geoResults = kMeans.geoGrab('1 VA Center', 'Augusta, ME')
geoResults['ResultSet']['Error']
geoResults['ResultSet']['Results'][0]['longitude']
geoResults['ResultSet']['Results'][0]['latitude']
# 转换类型为浮点数
kMeans.massPlaceFind('portlandClubs.txt')

# 绘图
from importlib import reload
reload(kMeans)
kMeans.clusterClubs(5)