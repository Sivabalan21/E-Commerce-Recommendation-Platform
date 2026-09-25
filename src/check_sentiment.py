import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("CheckSentiment") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

SENTIMENT_PATH = "hdfs:///user/ss21176_nyu_edu/ecommerce/sentiment/appliances/sentiment_data"

df = spark.read.parquet(SENTIMENT_PATH)

print(f"Row count     : {df.count():,}")
print(f"Columns       : {df.columns}")
print(f"Null check    :")
df.select([F.sum(F.when(F.col(c).isNull(),1).otherwise(0)).alias(c)
           for c in ["user_id","parent_asin","rating",
                     "review_text","sentiment_score","sentiment_label"]]).show()
print("Sample rows:")
df.select("user_id","parent_asin","rating",
          "review_text","sentiment_score","sentiment_label") \
  .show(3, truncate=60)

spark.stop()