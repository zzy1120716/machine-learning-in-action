import apriori
dataSet = apriori.loadDataSet()
dataSet
C1 = apriori.createC1(dataSet)
C1
D = list(map(set, dataSet))
D
L1, suppData0 = apriori.scanD(D, C1, 0.5)
L1

from importlib import reload
reload(apriori)
L, suppData = apriori.apriori(dataSet)
L
L[0]
L[1]
L[2]
L[3]
apriori.aprioriGen(L[0], 2)
L, suppData = apriori.apriori(dataSet, minSupport=0.7)
L

# 测试关联规则函数
reload(apriori)
L, suppData = apriori.apriori(dataSet, minSupport=0.5)
rules = apriori.generateRules(L, suppData, minConf=0.7)
rules
rules = apriori.generateRules(L, suppData, minConf=0.5)
rules

# 示例：发现毒蘑菇的相似特征
mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
L, suppData = apriori.apriori(mushDatSet, minSupport=0.3)
for item in L[1]:
    if item.intersection('2'):
        print(item)
for item in L[3]:
    if item.intersection('2'):
        print(item)