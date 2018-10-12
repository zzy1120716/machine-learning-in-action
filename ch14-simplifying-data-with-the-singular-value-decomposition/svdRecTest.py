from numpy import *
U, Sigma, VT = linalg.svd([[1, 1], [7, 7]])
U
Sigma
VT

import svdRec
Data = svdRec.loadExData()
U, Sigma, VT = linalg.svd(Data)
Sigma

# 重构原始矩阵
Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
U[:, :3] * Sig3 * VT[:3, :]

# 计算相似度
from importlib import reload
reload(svdRec)
myMat = mat(svdRec.loadExData())
# 欧氏距离
svdRec.ecludSim(myMat[:, 0], myMat[:, 4])
svdRec.ecludSim(myMat[:, 0], myMat[:, 0])
# 余弦相似度
svdRec.cosSim(myMat[:, 0], myMat[:, 4])
svdRec.cosSim(myMat[:, 0], myMat[:, 0])
# 皮尔逊相关系数
svdRec.pearsSim(myMat[:, 0], myMat[:, 4])
svdRec.pearsSim(myMat[:, 0], myMat[:, 0])

# 示例：餐馆菜肴推荐引擎
reload(svdRec)
myMat = mat(svdRec.loadExData())
myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
myMat[3, 3] = 2
myMat
svdRec.recommend(myMat, 2)
svdRec.recommend(myMat, 2, simMeas=svdRec.ecludSim)
svdRec.recommend(myMat, 2, simMeas=svdRec.pearsSim)

# 计算SVD
from numpy import linalg as la
U, Sigma, VT = la.svd(mat(svdRec.loadExData2()))
Sigma
Sig2 = Sigma ** 2
sum(Sig2)
sum(Sig2) * 0.9
sum(Sig2[: 2])
sum(Sig2[: 3])

# SVD评分估计
reload(svdRec)
myMat = mat(svdRec.loadExData2())
svdRec.recommend(myMat, 1, estMethod=svdRec.svdEst)
svdRec.recommend(myMat, 1, estMethod=svdRec.svdEst, simMeas=svdRec.pearsSim)

# 示例：基于SVD的图像压缩
reload(svdRec)
svdRec.imgCompress(2)
