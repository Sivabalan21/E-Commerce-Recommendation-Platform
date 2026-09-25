import sys
import time
import itertools
from functools import reduce
from pyspark import StorageLevel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

# ─────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────
if len(sys.argv) != 3:
    print("Usage: spark-submit 08_als_training.py <input_path> <output_path>")
    sys.exit(1)

INPUT_PATH  = sys.argv[1]
OUTPUT_BASE = sys.argv[2]

# ─────────────────────────────────────────
# CONFIG — Best params from HPO
# rank=50, regParam=0.1, maxIter=20
# RMSE=2.0469 (best of 27 combinations)
# ─────────────────────────────────────────
USE_GRID_SEARCH  = False
BEST_RANK        = 50
BEST_REG_PARAM   = 0.1
BEST_MAX_ITER    = 20

# ─────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("ALS_Final_Training") \
    .config("spark.sql.adaptive.enabled",      "true") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .config("spark.driver.memory",             "4g") \
    .config("spark.executor.memory",           "4g") \
    .config("spark.sql.shuffle.partitions",    "200") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

lines = []

def log(msg=""):
    lines.append(str(msg))
    print(msg)

def section(title):
    log()
    log("=" * 60)
    log(title)
    log("=" * 60)

def save_report():
    try:
        report_df = spark.createDataFrame([(l,) for l in lines], ["line"])
        report_df.coalesce(1).write.mode("overwrite").text(
            OUTPUT_BASE + "/report"
        )
        log("Report saved.")
    except Exception as e:
        log(f"Report save failed: {e}")

# ─────────────────────────────────────────
# STEP 1 — LOAD AND PREPARE
# ─────────────────────────────────────────
section("STEP 1 — LOADING AND PREPARING DATA")
log(f"Input  : {INPUT_PATH}")
log(f"Output : {OUTPUT_BASE}")
log(f"Config : rank={BEST_RANK} regParam={BEST_REG_PARAM} maxIter={BEST_MAX_ITER}")

df = spark.read.parquet(INPUT_PATH)

interactions = df.select(
    "user_id", "parent_asin", "rating"
).dropDuplicates(["user_id", "parent_asin"]) \
 .repartition(200)

interactions.persist(StorageLevel.MEMORY_AND_DISK)

interaction_count = interactions.count()

stats = interactions.agg(
    F.approx_count_distinct("user_id").alias("unique_users"),
    F.approx_count_distinct("parent_asin").alias("unique_products"),
    F.min("rating").alias("min_rating"),
    F.max("rating").alias("max_rating"),
    F.round(F.avg("rating"), 4).alias("avg_rating")
).collect()[0]

unique_users    = stats["unique_users"]
unique_products = stats["unique_products"]

log(f"Total interactions : {interaction_count:,}")
log(f"Unique users       : {unique_users:,}")
log(f"Unique products    : {unique_products:,}")
log(f"Matrix density     : {round(100*interaction_count/(unique_users*unique_products), 6)}%")
log(f"Rating range       : {stats['min_rating']} to {stats['max_rating']}, avg={stats['avg_rating']}")

# ─────────────────────────────────────────
# STEP 2 — STRING INDEXING
# ─────────────────────────────────────────
section("STEP 2 — STRING INDEXING")

log("Fitting StringIndexers in one pipeline pass...")

pipeline       = Pipeline(stages=[
    StringIndexer(inputCol="user_id",     outputCol="user_index"),
    StringIndexer(inputCol="parent_asin", outputCol="product_index")
])
pipeline_model = pipeline.fit(interactions)
df_indexed     = pipeline_model.transform(interactions) \
    .withColumn("user_index",    F.col("user_index").cast(IntegerType())) \
    .withColumn("product_index", F.col("product_index").cast(IntegerType())) \
    .select("user_index", "product_index", "rating")

df_indexed.persist(StorageLevel.MEMORY_AND_DISK)
interactions.unpersist()

idx_stats = df_indexed.agg(
    F.max("user_index").alias("max_user_idx"),
    F.max("product_index").alias("max_prod_idx")
).collect()[0]

log(f"user_index range    : 0 to {idx_stats['max_user_idx']:,}")
log(f"product_index range : 0 to {idx_stats['max_prod_idx']:,}")

user_indexer_model    = pipeline_model.stages[0]
product_indexer_model = pipeline_model.stages[1]

user_indexer_model.write().overwrite().save(OUTPUT_BASE + "/user_indexer")
product_indexer_model.write().overwrite().save(OUTPUT_BASE + "/product_indexer")
log("StringIndexer models saved.")

# ─────────────────────────────────────────
# STEP 3 — TRAIN TEST SPLIT
# ─────────────────────────────────────────
section("STEP 3 — TRAIN TEST SPLIT")

train_df, test_df = df_indexed.randomSplit([0.8, 0.2], seed=42)
train_df.persist(StorageLevel.MEMORY_AND_DISK)
test_df.persist(StorageLevel.MEMORY_AND_DISK)

log(f"Training rows (approx) : {int(interaction_count * 0.8):,}")
log(f"Test rows (approx)     : {int(interaction_count * 0.2):,}")

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

# ─────────────────────────────────────────
# STEP 4 — ALS TRAINING
# Using best params from HPO
# rank=50, regParam=0.1, maxIter=20
# ─────────────────────────────────────────
section("STEP 4 — ALS MODEL TRAINING")

log(f"Training ALS with HPO best params:")
log(f"  rank     : {BEST_RANK}")
log(f"  regParam : {BEST_REG_PARAM}")
log(f"  maxIter  : {BEST_MAX_ITER}")
log(f"  CV RMSE  : 2.0469 (best of 27 HPO combinations)")
log()

als = ALS(
    rank=BEST_RANK,
    regParam=BEST_REG_PARAM,
    maxIter=BEST_MAX_ITER,
    userCol="user_index",
    itemCol="product_index",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True
)

train_start = time.time()
best_model  = als.fit(train_df)
train_time  = round(time.time() - train_start, 1)

log(f"Training completed in {train_time}s")

# ─────────────────────────────────────────
# STEP 5 — SAVE MODEL IMMEDIATELY
# Save before any heavy post-processing
# so model is safe even if job is interrupted
# ─────────────────────────────────────────
section("STEP 5 — SAVING MODEL IMMEDIATELY")

best_model.write().overwrite().save(OUTPUT_BASE + "/als_model")
log(f"ALS model saved       : {OUTPUT_BASE}/als_model")

user_indexer_model.write().overwrite().save(OUTPUT_BASE + "/user_indexer")
product_indexer_model.write().overwrite().save(OUTPUT_BASE + "/product_indexer")
log(f"User indexer saved    : {OUTPUT_BASE}/user_indexer")
log(f"Product indexer saved : {OUTPUT_BASE}/product_indexer")

log()
log("Model is safely persisted. Proceeding with evaluation.")

save_report()

# ─────────────────────────────────────────
# STEP 6 — EVALUATION
# ─────────────────────────────────────────
section("STEP 6 — MODEL EVALUATION")

train_rmse = round(evaluator.evaluate(best_model.transform(train_df)), 4)
test_rmse  = round(evaluator.evaluate(best_model.transform(test_df)),  4)

mae_eval = RegressionEvaluator(
    metricName="mae", labelCol="rating", predictionCol="prediction"
)
test_mae = round(mae_eval.evaluate(best_model.transform(test_df)), 4)

log(f"Train RMSE : {train_rmse}")
log(f"Test RMSE  : {test_rmse}")
log(f"Test MAE   : {test_mae}")

save_report()

# ─────────────────────────────────────────
# STEP 7 — SAVE FACTOR MATRICES
# ─────────────────────────────────────────
section("STEP 7 — SAVING FACTOR MATRICES")

best_model.userFactors.write.mode("overwrite").parquet(
    OUTPUT_BASE + "/user_factors"
)
best_model.itemFactors.write.mode("overwrite").parquet(
    OUTPUT_BASE + "/item_factors"
)

user_count = best_model.userFactors.count()
item_count = best_model.itemFactors.count()

log(f"User factors saved    : {OUTPUT_BASE}/user_factors")
log(f"Item factors saved    : {OUTPUT_BASE}/item_factors")
log(f"User factors shape    : {user_count:,} x rank {BEST_RANK}")
log(f"Item factors shape    : {item_count:,} x rank {BEST_RANK}")

save_report()

# ─────────────────────────────────────────
# STEP 8 — GENERATE RECOMMENDATIONS
# ─────────────────────────────────────────
section("STEP 8 — GENERATING RECOMMENDATIONS")

log("Generating top 10 recommendations for all users...")
user_recs  = best_model.recommendForAllUsers(10)
user_count_recs = user_recs.count()
log(f"Users with recommendations : {user_count_recs:,}")

log()
log("Sample (2 users):")
for row in user_recs.limit(2).collect():
    recs = [(r.product_index, round(r.rating, 3)) for r in row["recommendations"]]
    log(f"  user_index={row['user_index']} -> {recs[:3]}...")

user_recs.write.mode("overwrite").parquet(
    OUTPUT_BASE + "/user_recommendations"
)
log(f"Recommendations saved : {OUTPUT_BASE}/user_recommendations")

save_report()

# ─────────────────────────────────────────
# STEP 9 — SAVE SUMMARY
# ─────────────────────────────────────────
section("STEP 9 — SAVING SUMMARY")

summary_data = [
    ("total_interactions",  str(interaction_count)),
    ("unique_users",        str(unique_users)),
    ("unique_products",     str(unique_products)),
    ("best_rank",           str(BEST_RANK)),
    ("best_regParam",       str(BEST_REG_PARAM)),
    ("best_maxIter",        str(BEST_MAX_ITER)),
    ("hpo_cv_rmse",         "2.0469"),
    ("train_rmse",          str(train_rmse)),
    ("test_rmse",           str(test_rmse)),
    ("test_mae",            str(test_mae)),
    ("users_with_recs",     str(user_count_recs)),
    ("training_time_s",     str(train_time)),
]

summary_df = spark.createDataFrame(summary_data, ["metric", "value"])
summary_df.coalesce(1).write.mode("overwrite") \
    .option("header", "true").csv(OUTPUT_BASE + "/summary")

log("Summary CSV saved.")

save_report()

log()
log("=" * 60)
log("ALS FINAL TRAINING COMPLETE")
log("=" * 60)
log(f"  rank={BEST_RANK}, regParam={BEST_REG_PARAM}, maxIter={BEST_MAX_ITER}")
log(f"  Train RMSE : {train_rmse}")
log(f"  Test RMSE  : {test_rmse}")
log(f"  Test MAE   : {test_mae}")

spark.stop()