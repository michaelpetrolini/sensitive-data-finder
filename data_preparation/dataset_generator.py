from pyspark.sql import SparkSession

from data_preparation.configs import RELIGION_PATH

spark = SparkSession.builder.appName('Genera SD dataset').getOrCreate()


religion_df = spark.sparkContext.emptyRDD()

religion_df = religion_df.union(spark.read.text(RELIGION_PATH + 'religion.txt').rdd)
religion_df.take(10)
print(religion_df.count())
