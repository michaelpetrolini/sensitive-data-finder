from pyspark.sql import SparkSession
from pyspark.sql.functions import length, rand

INPUT_PATH = "/user/73576/sdf/input/"
OUTPUT_PATH = "/user/srv-ldd/sdf/input/"
POLITICS_PATH = INPUT_PATH + "politics/"
MEDICAL_PATH = INPUT_PATH + "medical/"
FINANCE_PATH = INPUT_PATH + "finance/"
RELIGION_PATH = INPUT_PATH + "religion/"
SEXUALITY_PATH = INPUT_PATH + "sexuality/"
CLASSIFIED_PATH = OUTPUT_PATH + "classified/"
OTHER_PATH = INPUT_PATH + "other/"

spark = SparkSession.builder.appName('SD modeling').getOrCreate()

classified = spark.read.text(CLASSIFIED_PATH + "dataset.txt").withColumnRenamed('value', 'text')


def filter_and_save(dataframe, text):
    dataframe = dataframe.filter(length(dataframe.value) >= 100).filter(length(dataframe.value) <= 200).distinct()
    dataframe = dataframe.join(classified, dataframe.value == classified.text, "left")
    dataframe = dataframe.filter(dataframe.text.isNull()).select(dataframe.value)
    count = dataframe.count()
    dataframe = dataframe.sample(float(10000) / float(count)).orderBy(rand())
    dataframe.count()
    dataframe.rdd.flatMap(lambda x: x).map(lambda x: "\"\"\"" + x.replace("\"", '') + "\"\"\"").coalesce(1) \
        .saveAsTextFile(OUTPUT_PATH + text)
    return dataframe


politics = spark.read.text(POLITICS_PATH + "politics_aws.txt")
politics = filter_and_save(politics, "politics")

medical = spark.read.text(MEDICAL_PATH + "medical_aws.txt")
medical = filter_and_save(medical, "medical")

finance = spark.read.text(FINANCE_PATH + "finance_aws.txt")
finance = filter_and_save(finance, "finance")

religion = spark.read.text(RELIGION_PATH + "religion_aws.txt")
religion = filter_and_save(religion, "religion")

sexuality = spark.read.text(SEXUALITY_PATH + "sexuality_aws.txt")
sexuality = filter_and_save(sexuality, "sexuality")

df = politics.union(medical).union(finance).union(religion).union(sexuality)
df = df.orderBy(rand())
df.count()
df.rdd.flatMap(lambda x: x).map(lambda x: "\"\"\"" + x.replace("\"", '') + "\"\"\"").coalesce(1) \
    .saveAsTextFile(OUTPUT_PATH + "total")
