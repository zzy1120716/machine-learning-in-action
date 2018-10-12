from numpy import *
import pca
dataMat = pca.loadDataSet('testSet.txt')
# lowDMat, reconMat = pca.pca(dataMat, 1)
lowDMat, reconMat = pca.pca(dataMat, 2)
shape(lowDMat)

import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=50, c='blue')
ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
plt.show()

# 替换为平均值
from importlib import reload
reload(pca)
dataMat = pca.replaceNanWithMean()
meanVals = mean(dataMat, axis=0)
meanRemoved = dataMat - meanVals
covMat = cov(meanRemoved, rowvar=0)
eigVals, eigVects = linalg.eig(mat(covMat))
eigVals