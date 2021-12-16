from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, IndexToString, StringIndexer, Tokenizer
from pyspark.mllib.evaluation import MulticlassMetrics

from data_preparation import df_modeling


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


train_df, test_df = df_modeling.get_df()
pipeline = build_pipeline(train_df)

model = pipeline.fit(train_df)
predictions = model.transform(test_df)


# Overall statistics
predictionAndLabels = predictions.rdd.map(lambda x: (x.prediction, x.indexedLabel))
metrics = MulticlassMetrics(predictionAndLabels)
precision = metrics.precision()
recall = metrics.recall()
f1Score = metrics.fMeasure()
print("Summary Stats")
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score)

# Statistics by class
labels = predictions.rdd.map(lambda x: (x.indexedLabel, x.label)).distinct().collect()
for label in sorted(labels):
    print("Class %s precision = %s" % (label[1], metrics.precision(label[0])))
    print("Class %s recall = %s" % (label[1], metrics.recall(label[0])))
    print("Class %s F1 Measure = %s" % (label[1], metrics.fMeasure(label[0], beta=1.0)))

# Weighted stats
print("Weighted recall = %s" % metrics.weightedRecall)
print("Weighted precision = %s" % metrics.weightedPrecision)
print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)
