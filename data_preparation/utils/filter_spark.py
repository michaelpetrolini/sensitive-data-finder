from pyspark.sql.functions import length, rand

INPUT_PATH = "/user/73576/sdf/input/"
OUTPUT_PATH = "/user/srv-ldd/sdf/input/"
POLITICS_PATH = INPUT_PATH + "politics/"
MEDICAL_PATH = INPUT_PATH + "medical/"
FINANCE_PATH = INPUT_PATH + "finance/"
RELIGION_PATH = INPUT_PATH + "religion/"
SEXUALITY_PATH = INPUT_PATH + "sexuality/"

spark = SparkSession.builder.appName('SD modeling').getOrCreate()

politics = spark.read.text(POLITICS_PATH + "politics_aws.txt")
politics = politics.filter(length(politics.value) >= 100).filter(length(politics.value) <= 200).distinct()
count = politics.count()
politics = politics.sample(float(10000)/float(count)).orderBy(rand())
politics.count()
politics.rdd.flatMap(lambda x: x).coalesce(1).saveAsTextFile(OUTPUT_PATH + "politics")

medical = spark.read.text(MEDICAL_PATH + "medical_aws.txt")
medical = medical.filter(length(medical.value) >= 100).filter(length(medical.value) <= 200).distinct()
count = medical.count()
medical = medical.sample(float(10000)/float(count)).orderBy(rand())
medical.count()
medical.rdd.flatMap(lambda x: x).coalesce(1).saveAsTextFile(OUTPUT_PATH + "medical")

finance = spark.read.text(FINANCE_PATH + "finance_aws.txt")
finance = finance.filter(length(finance.value) >= 100).filter(length(finance.value) <= 200).distinct()
count = finance.count()
finance = finance.sample(float(10000)/float(count)).orderBy(rand())
finance.count()
finance.rdd.flatMap(lambda x: x).coalesce(1).saveAsTextFile(OUTPUT_PATH + "finance")

religion = spark.read.text(RELIGION_PATH + "religion_aws.txt")
religion = religion.filter(length(religion.value) >= 100).filter(length(religion.value) <= 200).distinct()
count = religion.count()
religion = religion.sample(float(10000)/float(count)).orderBy(rand())
religion.count()
religion.rdd.flatMap(lambda x: x).coalesce(1).saveAsTextFile(OUTPUT_PATH + "religion")

sexuality = spark.read.text(SEXUALITY_PATH + "sexuality_aws.txt")
sexuality = sexuality.filter(length(sexuality.value) >= 100).filter(length(sexuality.value) <= 200).distinct()
count = sexuality.count()
sexuality = sexuality.sample(float(10000)/float(count)).orderBy(rand())
sexuality.count()
sexuality.rdd.flatMap(lambda x: x).coalesce(1).saveAsTextFile(OUTPUT_PATH + "sexuality")

df = politics.union(medical).union(finance).union(religion).union(sexuality)
df = df.orderBy(rand())
df.count()
df.rdd.flatMap(lambda x: x).coalesce(1).saveAsTextFile(OUTPUT_PATH + "total")
