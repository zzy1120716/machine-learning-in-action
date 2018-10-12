import fpGrowth
rootNode = fpGrowth.treeNode('pyramid', 9, None)
rootNode.children['eye'] = fpGrowth.treeNode('eye', 13, None)
rootNode.disp()
rootNode.children['phoenix'] = fpGrowth.treeNode('phoenix', 3, None)
rootNode.disp()

from importlib import reload
reload(fpGrowth)
simpDat = fpGrowth.loadSimpDat()
simpDat
initSet = fpGrowth.createInitSet(simpDat)
initSet
# 创建FP树
myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 3)
myFPtree.disp()

reload(fpGrowth)
fpGrowth.findPrefixPath('x', myHeaderTab['x'][1])
fpGrowth.findPrefixPath('z', myHeaderTab['z'][1])
fpGrowth.findPrefixPath('r', myHeaderTab['r'][1])

reload(fpGrowth)
freqItems = []
fpGrowth.mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
freqItems

# 示例：从新闻网站点击流中挖掘
parsedDat = [line.split() for line in open('kosarak.dat').readlines()]
initSet = fpGrowth.createInitSet(parsedDat)
myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 100000)
myFreqList = []
fpGrowth.mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)
len(myFreqList)
myFreqList