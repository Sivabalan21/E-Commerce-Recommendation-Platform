import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("DataCheck") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

REVIEWS_PATH  = "hdfs:///user/ss21176_nyu_edu/ecommerce/cleaning/appliances/reviews/cleaned_reviews"
METADATA_PATH = "hdfs:///user/ss21176_nyu_edu/ecommerce/cleaning/appliances/metadata/cleaned_metadata"

print("\n" + "="*60)
print("REVIEWS CHECK")
print("="*60)

df_r = spark.read.parquet(REVIEWS_PATH)
print(f"Row count  : {df_r.count():,}")
print(f"Columns    : {df_r.columns}")
df_r.select([F.sum(F.when(F.col(c).isNull(),1).otherwise(0)).alias(c) for c in df_r.columns]).show()
df_r.show(3, truncate=80)

print("\n" + "="*60)
print("METADATA CHECK")
print("="*60)

df_m = spark.read.parquet(METADATA_PATH)
print(f"Row count  : {df_m.count():,}")
print(f"Columns    : {df_m.columns}")
df_m.select([F.sum(F.when(F.col(c).isNull(),1).otherwise(0)).alias(c) for c in df_m.columns]).show()
df_m.show(3, truncate=80)

spark.stop()