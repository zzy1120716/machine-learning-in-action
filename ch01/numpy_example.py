from numpy import *
random.rand(4, 4)   # 随机生成4 * 4数组
randMat = mat(random.rand(4, 4))    # 数组转化为矩阵
invRandMat = randMat.I   # 矩阵求逆
myEye = randMat * invRandMat
myEye - eye(4)  # eye(4)创建4 * 4的单位矩阵
