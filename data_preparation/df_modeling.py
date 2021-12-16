from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

from data_preparation import RELIGION_PATH, POLITICS_PATH, MEDICAL_PATH, FINANCE_PATH, SEXUALITY_PATH

TRAIN_WEIGHT = 0.8
SPLIT_SEED = 0
SAMPLE_SEED = 0

spark = SparkSession.builder.appName('SD modeling').getOrCreate()


def retrieve(path, label):
    return spark.read.text(path).rdd.flatMap(lambda x: x).map(lambda x: (x, label)).toDF(['text', 'label'])


def sample_and_split(df, numerator, denominator):
    return df.sample(float(numerator)/float(denominator), SAMPLE_SEED)\
        .randomSplit([TRAIN_WEIGHT, 1 - TRAIN_WEIGHT], SPLIT_SEED)


def get_df():
    religion_df = retrieve(RELIGION_PATH + 'religion_clean.txt', 'R')
    religion_count = religion_df.count()
    politics_df = retrieve(POLITICS_PATH + 'politics_clean.txt', 'P')
    politics_count = politics_df.count()
    medical_df = retrieve(MEDICAL_PATH + 'medical_clean.txt', 'M')
    medical_count = medical_df.count()
    finance_df = retrieve(FINANCE_PATH + 'finance_clean.txt', 'F')
    finance_count = finance_df.count()
    sexuality_df = retrieve(SEXUALITY_PATH + 'sexuality_clean.txt', 'S')
    sexuality_count = sexuality_df.count()
    minimum = min(religion_count, politics_count, medical_count, finance_count, sexuality_count)
    religion_train, religion_test = sample_and_split(religion_df, minimum, religion_count)
    politics_train, politics_test = sample_and_split(politics_df, minimum, politics_count)
    medical_train, medical_test = sample_and_split(medical_df, minimum, medical_count)
    finance_train, finance_test = sample_and_split(finance_df, minimum, finance_count)
    sexuality_train, sexuality_test = sample_and_split(sexuality_df, minimum, sexuality_count)
    train_df = religion_train.union(politics_train).union(medical_train).union(finance_train).union(sexuality_train). \
        withColumn('id', monotonically_increasing_id())
    test_df = religion_test.union(politics_test).union(medical_test).union(finance_test).union(sexuality_test). \
        withColumn('id', monotonically_increasing_id())
    return train_df, test_df


def set_labels(df, label):
    def map_labels(x):
        lbl = label if x.label == label else 'X'
        return lbl
    return df.rdd.map(lambda x: (x.id, x.text, map_labels(x))).toDF(['id', 'text', 'label'])
