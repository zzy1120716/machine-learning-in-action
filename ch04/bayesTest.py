# 测试创建词表
import bayes
listOPosts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
myVocabList
bayes.setOfWords2Vec(myVocabList, listOPosts[0])
bayes.setOfWords2Vec(myVocabList, listOPosts[3])