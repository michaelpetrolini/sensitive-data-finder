from pyspark.sql import SparkSession

from data_preparation.df_modeling import get_df, set_labels
from data_preparation.random_forest import build_pipeline
import pyspark.sql.functions as func
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType


def add_prediction(x):
    labels = x.predictions
    if x.predictedLabel != 'X':
        labels.append(x.predictedLabel)
    return [x.id, x.text, x.label, labels]


spark = SparkSession.builder.appName('SD modeling').getOrCreate()
LABELS = ['R', 'P', 'M', 'F', 'S']

train_df, test_df = get_df()

test_df = test_df.withColumn('predictions', func.array())

for label in LABELS:
    labeled_train = set_labels(train_df, label)
    labeled_test = set_labels(test_df, label)
    pipeline = build_pipeline(labeled_train)
    model = pipeline.fit(labeled_train)
    predictions = model.transform(labeled_test).select('id', 'predictedLabel')
    schema = StructType([StructField('id', LongType(), False),
                         StructField('text', StringType(), True),
                         StructField('label', StringType(), True),
                         StructField('predictions', ArrayType(StringType()), True)])
    test_df = spark.createDataFrame(test_df.join(predictions, 'id').rdd
                                    .map(lambda x: add_prediction(x)), schema)

test_df = test_df.withColumn('correct', func.expr('array_contains(predictions, label)'))

for label in LABELS:
    n_correct = test_df.filter(test_df["label"] == label).filter(test_df['correct']).count()
    n_total = test_df.filter(test_df["label"] == label).count()
    print('model accuracy for label ' + label + ': ' + str(float(n_correct) / float(n_total)))

n_correct = test_df.filter(test_df['correct']).count()
n_total = test_df.count()

print('model accuracy: ' + str(float(n_correct)/float(n_total)))
