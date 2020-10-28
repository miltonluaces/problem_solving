from pyspark import SparkContext

def WordCount():
    fileName = "D:/ProgramFiles/Spark/spark-2.3.2-bin-hadoop2.7/README.md"  
    sc = SparkContext("local", "first app")
    data = sc.textFile(fileName).cache()
    countA = data.filter(lambda s: 'a' in s).count()
    countB = data.filter(lambda s: 'b' in s).count()
    print('\nResult'); print('Lines with a: ', countA , 'lines with b: ', countB); print('')

def RDD():
    sc = SparkContext("local", "first app")
    data = sc.parallelize([('Amber', 22), ('Alfred', 23), ('Skye',4), ('Albert', 12),('Amber', 9)])


#WordCount()
RDD()