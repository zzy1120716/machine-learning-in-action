# Mapper
## Linux
cat inputFile.txt | python mrMeanMapper.py
## Windows
python mrMeanMapper.py < inputFile.txt

# Reducer
## Linux
cat inputFile.txt | python mrMeanMapper.py | python mrMeanReducer.py
## Windows
python mrMeanMapper.py < inputFile.txt | python mrMeanReducer.py

# 运行MapReduce任务
hadoop fs -copyFromLocal inputFile.txt mrmean-i
hadoop jar $HADOOP_HOME/contrib/streaming/hadoop-0.20.2-streaming.jar \
-input mrmean-i -output mrmean-o \
-mapper "python mrMeanMapper.py" \
-reducer "python mrMeanReducer.py"
hadoop fs -cat mrmean-o/part-00000
hadoop fs -copyToLocal mrmean-o/part-00000

# mrjob与EMR的无缝集成
## 单机
python mrMean.py < inputFile.txt > myOut.txt
## EMR上
python mrMean.py -r emr < inputFile.txt > myOut.txt
## 分步骤执行
python mrMean.py --mapper < inputFile.txt
## 整个运行
python mrMean.py < inputFile.txt
python mrMean.py -r emr < inputFile.txt > outFile.txt

## 本机
python mrSVM.py < kickStart.txt
## EMR
python mrSVM.py -r emr --num-ec2-instances=3 < kickStart.txt > myLog.txt
## 查看所有运行参数
python mrSVM.py -h