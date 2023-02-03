#import all needed libraries

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover
from pyspark.sql import SparkSession

import time

#create Spark session
appName = "Sentiment Analysis in Spark"
spark = SparkSession \
    .builder \
    .appName(appName) \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
    
tweets_data_csv = spark.read.csv('dataset/tweets.csv', inferSchema=True, header=True)
tweets_data_csv.show(truncate=False, n=3)

#select only "SentimentText" and "Sentiment" column and also cast "Sentiment" column data into integer with col name label
selected_data = tweets_data_csv.select("SentimentText", col("Sentiment").cast("Int").alias("label"))
selected_data.show(truncate = False,n=5)

#divide the data into 70% for training and 30% for testing
train_test_data = selected_data.randomSplit([0.7, 0.3]) 
trainingData = train_test_data[0] # index 0 for data training
testingData = train_test_data[1] #index 1  for data testing
train_rows = trainingData.count() #count training data
test_rows = testingData.count() #count testing data
print ("Training data rows:", train_rows, "; Testing data rows:", test_rows )

#Separate SentimentText coloumn into individual words using tokenizer (BAG OF WORDS)
# To extract features from text documents.
tokens = Tokenizer(inputCol="SentimentText", outputCol="SentimentWords")
tokens_Train = tokens.transform(trainingData)
tokens_Train.show(truncate=False, n=5)


#Stopwords are words which do not contain enough significance to be used without our algorithm.
#examples I,The,is,a
stop_words = StopWordsRemover(inputCol=tokens.getOutputCol(), 
                       outputCol="MeaningfulWords")
stop_words_train = stop_words.transform(tokens_Train)
stop_words_train.show(truncate=False, n=5)


#HashingTF converts documents into a numerical representation which can be fed directly or with further processing into other algorithms
#HashingTF converts documents to vectors of fixed size. 
#The terms are mapped to indices using a Hash Function. 
#The hash function used is MurmurHash 
#The term frequencies are computed with respect to the mapped indices.

hashTF = HashingTF(inputCol=stop_words.getOutputCol(), outputCol="features")
train_numeric = hashTF.transform(stop_words_train).select(
    'label', 'MeaningfulWords', 'features')
train_numeric.show(truncate=False, n=3)

#Train our classifier model using training data

lr = LogisticRegression(labelCol="label", featuresCol="features", 
                        maxIter=10, regParam=0.01)
model = lr.fit(train_numeric)
print ("-----Training is done!")


#Repeats the above steps for testing data
# step1: tokenize the data
# step2: remove stop words
# step 3: convert the text data into numeric data (extract features)

tokens_Test = tokens.transform(testingData)
stop_words_test = stop_words.transform(tokens_Test)
numericTest = hashTF.transform(stop_words_test).select(
    'Label', 'MeaningfulWords', 'features')
numericTest.show(truncate=False, n=4)



#Calculate the run time taken
#Calculate the accuracy


start_time=time.time()
prediction = model.transform(numericTest)
prediction_data = prediction.select(
    "MeaningfulWords", "prediction", "Label")
time_taken =time.time()-start_time
prediction_data.show(n=4, truncate = False)
right_Prediction = prediction_data.filter(
    prediction_data['prediction'] == prediction_data['Label']).count()
data_count = prediction_data.count()

print("correct prediction:", right_Prediction, ", total data:", data_count, 
      ", accuracy:", right_Prediction/data_count)
print("Time Taken",time_taken)



