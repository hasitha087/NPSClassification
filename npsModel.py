import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

from pyspark.sql import SparkSession, Row, functions, types
from pyspark.sql.functions import udf
import numpy as np
import pandas as pd
from pandas import DataFrame

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, VectorAssembler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.classification import NaiveBayes
from pyspark.mllib.evaluation import MultilabelMetrics
from pyspark.mllib.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from pyspark.rdd import RDD

spark = SparkSession.builder\
    .master("local")\
    .appName("Naive_Bayes-Multi-label-Classification Model")\
    .enableHiveSupport()\
    .getOrCreate()

log4j = spark._jvm.org.apache.log4j
log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)

lemmatizer = WordNetLemmatizer()

#############Lemmatize/Stemminize######################
def lemmaStemma(text):
    return lemmatizer.lemmatize(text)

############Multi-label Classification#################
def classifier():
    tokenizer = Tokenizer(inputCol="answer", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=900000)
    nb = NaiveBayes(smoothing=3.0, modelType="multinomial")
    pipeline = Pipeline(stages=[tokenizer, hashingTF, nb])
    return pipeline

############Model Testing###############################
def getPredictionsLabels(model, test_data):
    predictions = model.predict(test_data.map(lambda r: r.features))
    return predictions.zip(test_data.map(lambda r: r.label))

##############Evaluation Matrices########################	
def printMetrics(predictions_and_labels):
    metrics = MulticlassMetrics(predictions_and_labels)
    print ('Precision of True ', metrics.precision(1))
    print ('Precision of False', metrics.precision(0))
    print ('Recall of True    ', metrics.recall(1))
    print ('Recall of False   ', metrics.recall(0))
    print ('F-1 Score         ', metrics.fMeasure())
    print ('Confusion Matrix\n', metrics.confusionMatrix().toArray())

##########Split Probability array into multiple columns#####
def extract(row):
    return (row.answer,row.label,row.prediction,) + tuple(row.probability.toArray().tolist())

	
def main():
     
    csvData = spark.sql("select answer,label from training_table")
    dataset = csvData.dropna()

    dataset = dataset.toPandas()
    dataset['answer'] = dataset.answer.apply(lemmaStemma)
    dataset = spark.createDataFrame(dataset)

    train_data, test_data = dataset.randomSplit([0.7, 0.3])

    model = classifier().fit(train_data)
    prData = model.transform(test_data)
    clasDataFrame = prData.toPandas()

    evaluatorRecall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    recall = evaluatorRecall.evaluate(prData)
    print("Recall %s" % recall)

    ####Cross Validation####
    paramGrid = ParamGridBuilder().build()
    cv = CrossValidator()\
        .setEstimator(classifier())\
        .setEvaluator(evaluatorRecall)\
        .setEstimatorParamMaps(paramGrid)

    cvModel = cv.fit(train_data)
    cvPredictions = cvModel.transform(test_data)

    predictionsAndLabels = cvPredictions.select("prediction","label").rdd

    metrics = MulticlassMetrics(predictionsAndLabels)
    metricsAUC = BinaryClassificationMetrics(predictionsAndLabels)
    print("Cross Validated Confusion Matrix\n = %s" % metrics.confusionMatrix().toArray())
    print("Cross Validated recall = %s" % metrics.weightedRecall)
    print("Cross Validated Precision = %s" % metrics.weightedPrecision)
    print("Cross Validated fMeasure = %s" % metrics.weightedFMeasure)
    print("Cross Validated Accuracy = %s" % metrics.accuracy)
    print("Cross Validated AUC = %s" % metricsAUC.areaUnderROC)
	
    cvPred = cvPredictions.select("answer","label","probability","prediction")

    output = cvPred.rdd.map(extract).toDF(["answer","label","prediction"])
   
    ####Model Save####
    bestModel = cvModel.bestModel
    bestModel.write().overwrite().save("hdfs://nameservice1//source/NPSClassification")

if __name__ == '__main__':
    main()
