# 测试创建词表
import bayes
listOPosts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
myVocabList
bayes.setOfWords2Vec(myVocabList, listOPosts[0])
bayes.setOfWords2Vec(myVocabList, listOPosts[3])

# 测试朴素贝叶斯分类器训练函数
from numpy import *
from importlib import reload
reload(bayes)
listOPosts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
pAb
p0V

# 测试朴素贝叶斯分类器
reload(bayes)
bayes.testingNB()

# 朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 测试split方法
mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
mySent.split()
# 使用正则去除标点
import re
regEx = re.compile('\\W*')
listOfTokens = regEx.split(mySent)
listOfTokens
# 去掉空串
[tok for tok in listOfTokens if len(tok) > 0]
# 转换为小写
[tok.lower() for tok in listOfTokens if len(tok) > 0]
# 实际处理电子邮件
emailText = open('email/ham/6.txt').read()
listOfTokens = regEx.split(emailText)
listOfTokens

# 测试垃圾邮件过滤
reload(bayes)
bayes.spamTest()

"""
安装feedparser
git clone https://github.com/kurtmckee/feedparser.git
cd feedparser
python setup.py install
"""
# 打开Craigslist上的RSS源
import feedparser
ny = feedparser.parse('https://newyork.craigslist.org/search/res?format=rss')
ny['entries']
len(ny['entries'])

# 测试RSS源分类器
reload(bayes)
ny = feedparser.parse('https://newyork.craigslist.org/search/res?format=rss')
sf = feedparser.parse('https://sfbay.craigslist.org/search/res?format=rss')
vocabList, pSF, pNY = bayes.localWords(ny, sf)
vocabList, pSF, pNY = bayes.localWords(ny, sf)
# 显示词汇
reload(bayes)
bayes.getTopWords(ny, sf)
