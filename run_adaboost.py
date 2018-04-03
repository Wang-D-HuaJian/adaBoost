import sys
sys.path.append("F:/pythonTest")
from adaboost_meta_algorithm.adaboost import *

#dataMat,classLables = loadSimpData()
#print(dataMat)
#D = mat(ones((5,1))/5)
#buildStump(dataMat, classLables, D)
dataMat,labelArr = loadDataSet('horseColicTraining2.txt')
classifierArray,aggClassEst = adaBoostTrainDS(dataMat,labelArr,10)
plotROC(aggClassEst.T,labelArr)
#testArr,testLabelArr = loadDataSet('horseColicTest2.txt')
#prediction10 = adaClassify(testArr,classifierArray)
#print(prediction10)
#errArr=mat(ones((67,1)))
#num_error=errArr[prediction10!=mat(testLabelArr).T].sum()
#print(num_error/67)
#data=[]
#fr=open('test.txt')
#for line in fr.readlines():
#    data.append((line.strip()))
#print(data)
#for i in data:
    #print(i)
#    print(adaClassify( i,classifierArray))