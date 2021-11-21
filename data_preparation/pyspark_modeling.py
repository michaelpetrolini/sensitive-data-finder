from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, CountVectorizer, StringIndexer, IndexToString
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession

from data_preparation import RELIGION_PATH, POLITICS_PATH, MEDICAL_PATH, FINANCE_PATH, SEXUALITY_PATH

TRAIN_WEIGHT = 0.8
SPLIT_SEED = 0


def build_pipeline(df):
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashing_tf = HashingTF(inputCol="words", outputCol="rawFeatures")
    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=100)
    # alternatively, CountVectorizer can also be used to get term frequency vectors
    cv = CountVectorizer(inputCol="words", outputCol="features")
    label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df)
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features")
    label_converter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=label_indexer.labels)
    return Pipeline(stages=[tokenizer, cv, label_indexer, rf, label_converter])


spark = SparkSession.builder.appName('SD modeling').getOrCreate()

religion_train, religion_test = spark.read.text(RELIGION_PATH + 'religion_clean.txt').rdd \
    .flatMap(lambda x: x).map(lambda x: (x, 'R')).toDF(['text', 'label']) \
    .randomSplit([TRAIN_WEIGHT, 1 - TRAIN_WEIGHT], SPLIT_SEED)

politics_train, politics_test = spark.read.text(POLITICS_PATH + 'politics_clean.txt').rdd \
    .flatMap(lambda x: x).map(lambda x: (x, 'P')).toDF(['text', 'label']) \
    .randomSplit([TRAIN_WEIGHT, 1 - TRAIN_WEIGHT], SPLIT_SEED)

medical_train, medical_test = spark.read.text(MEDICAL_PATH + 'medical_clean.txt').rdd \
    .flatMap(lambda x: x).map(lambda x: (x, 'M')).toDF(['text', 'label']) \
    .randomSplit([TRAIN_WEIGHT, 1 - TRAIN_WEIGHT], SPLIT_SEED)

finance_train, finance_test = spark.read.text(FINANCE_PATH + 'finance_clean.txt').rdd \
    .flatMap(lambda x: x).map(lambda x: (x, 'F')).toDF(['text', 'label']) \
    .randomSplit([TRAIN_WEIGHT, 1 - TRAIN_WEIGHT], SPLIT_SEED)

sexuality_train, sexuality_test = spark.read.text(SEXUALITY_PATH + 'sexuality_clean.txt').rdd \
    .flatMap(lambda x: x).map(lambda x: (x, 'S')).toDF(['text', 'label']) \
    .randomSplit([TRAIN_WEIGHT, 1 - TRAIN_WEIGHT], SPLIT_SEED)

train_df = religion_train.union(politics_train).union(medical_train).union(finance_train).union(sexuality_train)
test_df = religion_test.union(politics_test).union(medical_test).union(finance_test).union(sexuality_test)

pipeline = build_pipeline(train_df)

model = pipeline.fit(train_df)
predictions = model.transform(test_df)


predictionAndLabels = predictions.select('prediction', 'indexedLabel')\
    .rdd.map(lambda x: (x['prediction'], x['indexedLabel']))
metrics = MulticlassMetrics(predictionAndLabels)
precision = metrics.precision()
recall = metrics.recall()
f1Score = metrics.fMeasure()
print("Summary Stats")
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score)
