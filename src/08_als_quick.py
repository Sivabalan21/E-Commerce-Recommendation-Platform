import sys
import time
import itertools
from functools import reduce
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer

# ─────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────
if len(sys.argv) != 3:
    print("Usage: spark-submit 08_als_training.py <input_path> <output_path>")
    print("Example:")
    print("  spark-submit 08_als_training.py \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/sentiment/appliances/sentiment_data \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/models/appliances")
    sys.exit(1)

INPUT_PATH  = sys.argv[1]
OUTPUT_BASE = sys.argv[2]

# ─────────────────────────────────────────
# CONFIG
# USE_GRID_SEARCH = True  → manual grid search with logging
# USE_GRID_SEARCH = False → single run with default params
# ─────────────────────────────────────────
USE_GRID_SEARCH     = False
GRID_FOLDS          = 2
GRID_RANKS          = [10, 20]
GRID_REG_PARAMS     = [0.01, 0.1]
GRID_MAX_ITERS      = [10, 20]

# ─────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("ALS_Training") \
    .config("spark.sql.adaptive.enabled",      "true") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .config("spark.driver.memory",             "4g") \
    .config("spark.executor.memory",           "4g") \
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

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
section("STEP 1 — LOADING SENTIMENT DATA")

df = spark.read.parquet(INPUT_PATH)
df.cache()

total_rows = df.count()
log(f"Total rows     : {total_rows:,}")
log(f"Columns        : {df.columns}")

# ─────────────────────────────────────────
# BUILD INTERACTION MATRIX
# ─────────────────────────────────────────
section("STEP 2 — BUILDING INTERACTION MATRIX")

interactions = df.select(
    "user_id",
    "parent_asin",
    "rating"
).distinct()

interactions.cache()
interaction_count = interactions.count()

unique_users    = interactions.select(
    F.approx_count_distinct("user_id")).collect()[0][0]
unique_products = interactions.select(
    F.approx_count_distinct("parent_asin")).collect()[0][0]

log(f"Total interactions : {interaction_count:,}")
log(f"Unique users       : {unique_users:,}")
log(f"Unique products    : {unique_products:,}")
log(f"Matrix density     : {round(100*interaction_count/(unique_users*unique_products), 6)}%")

log()
log(f"{'Rating':<10} {'Count':>12} {'Percentage':>12}")
log("-" * 36)
for row in interactions.groupBy("rating").count().orderBy("rating").collect():
    pct = round(100 * row["count"] / interaction_count, 2)
    log(f"{row['rating']:<10} {row['count']:>12,} {pct:>11}%")

# ─────────────────────────────────────────
# STRING INDEXING
# ─────────────────────────────────────────
section("STEP 3 — STRING INDEXING")

log("Fitting user StringIndexer...")
user_indexer       = StringIndexer(inputCol="user_id",     outputCol="user_index")
user_indexer_model = user_indexer.fit(interactions)
df_indexed         = user_indexer_model.transform(interactions)

log("Fitting product StringIndexer...")
product_indexer       = StringIndexer(inputCol="parent_asin",  outputCol="product_index")
product_indexer_model = product_indexer.fit(df_indexed)
df_indexed            = product_indexer_model.transform(df_indexed)

df_indexed = df_indexed \
    .withColumn("user_index",    F.col("user_index").cast(IntegerType())) \
    .withColumn("product_index", F.col("product_index").cast(IntegerType()))

df_indexed.cache()

log(f"user_index range    : 0 to {df_indexed.agg(F.max('user_index')).collect()[0][0]:,}")
log(f"product_index range : 0 to {df_indexed.agg(F.max('product_index')).collect()[0][0]:,}")

user_indexer_model.write().overwrite().save(OUTPUT_BASE + "/user_indexer")
product_indexer_model.write().overwrite().save(OUTPUT_BASE + "/product_indexer")
log("StringIndexer models saved.")

# ─────────────────────────────────────────
# TRAIN TEST SPLIT
# ─────────────────────────────────────────
section("STEP 4 — TRAIN TEST SPLIT")

train_df, test_df = df_indexed.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()

train_count = train_df.count()
test_count  = test_df.count()

log(f"Training rows : {train_count:,}  ({round(100*train_count/interaction_count,2)}%)")
log(f"Test rows     : {test_count:,}  ({round(100*test_count/interaction_count,2)}%)")

# ─────────────────────────────────────────
# EVALUATOR
# ─────────────────────────────────────────
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

# ─────────────────────────────────────────
# GRID SEARCH WITH LOGGING
# ─────────────────────────────────────────
section("STEP 5 — ALS HYPERPARAMETER SEARCH")

if USE_GRID_SEARCH:
    param_grid = list(itertools.product(
        GRID_RANKS,
        GRID_REG_PARAMS,
        GRID_MAX_ITERS
    ))

    total_trials = len(param_grid)
    log(f"Search space:")
    log(f"  rank     : {GRID_RANKS}")
    log(f"  regParam : {GRID_REG_PARAMS}")
    log(f"  maxIter  : {GRID_MAX_ITERS}")
    log(f"  Total combinations : {total_trials}")
    log(f"  Folds per trial    : {GRID_FOLDS}")
    log(f"  Total ALS runs     : {total_trials * GRID_FOLDS}")
    log()

    best_rmse     = float("inf")
    best_params   = None
    trial_results = []
    job_start     = time.time()

    for i, (rank, reg_param, max_iter) in enumerate(param_grid):
        trial_start = time.time()
        log(f"[{i+1:>2}/{total_trials}] rank={rank:<4} regParam={reg_param:<6} maxIter={max_iter:<4} starting...")

        fold_rmse  = []
        folds_data = df_indexed.randomSplit([1.0/GRID_FOLDS]*GRID_FOLDS, seed=42)

        for fold in range(GRID_FOLDS):
            val_fold   = folds_data[fold]
            train_fold = reduce(
                DataFrame.union,
                [folds_data[j] for j in range(GRID_FOLDS) if j != fold]
            )

            als_trial = ALS(
                rank=rank,
                regParam=reg_param,
                maxIter=max_iter,
                userCol="user_index",
                itemCol="product_index",
                ratingCol="rating",
                coldStartStrategy="drop",
                nonnegative=True,
                implicitPrefs=False
            )

            model_cv  = als_trial.fit(train_fold)
            preds_cv  = model_cv.transform(val_fold)
            rmse_fold = round(evaluator.evaluate(preds_cv), 4)
            fold_rmse.append(rmse_fold)
            log(f"  fold {fold+1}/{GRID_FOLDS} RMSE={rmse_fold}")

        avg_rmse     = round(sum(fold_rmse) / len(fold_rmse), 4)
        elapsed      = round(time.time() - trial_start, 1)
        total_elapsed = round((time.time() - job_start) / 60, 1)
        trial_results.append((rank, reg_param, max_iter, avg_rmse))

        log(f"  avg RMSE={avg_rmse}  trial_time={elapsed}s  total_elapsed={total_elapsed}min")

        if avg_rmse < best_rmse:
            best_rmse   = avg_rmse
            best_params = (rank, reg_param, max_iter)
            log(f"  *** New best! RMSE={best_rmse} params=rank:{rank} reg:{reg_param} iter:{max_iter} ***")
        log()

    best_rank, best_reg, best_maxIter = best_params

    log("=" * 60)
    log("GRID SEARCH COMPLETE")
    log("=" * 60)
    log(f"Best rank     : {best_rank}")
    log(f"Best regParam : {best_reg}")
    log(f"Best maxIter  : {best_maxIter}")
    log(f"Best avg RMSE : {best_rmse}")
    log()
    log("All trial results:")
    log(f"{'#':<5} {'rank':<8} {'regParam':<12} {'maxIter':<10} {'avg_RMSE':>10}")
    log("-" * 48)
    for idx, (r, rp, mi, rmse) in enumerate(trial_results):
        marker = " <-- BEST" if (r, rp, mi) == best_params else ""
        log(f"{idx+1:<5} {r:<8} {rp:<12} {mi:<10} {rmse:>10}{marker}")

    log()
    log(f"Training final model with best params on full training set...")
    final_als = ALS(
        rank=best_rank,
        regParam=best_reg,
        maxIter=best_maxIter,
        userCol="user_index",
        itemCol="product_index",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        implicitPrefs=False
    )
    best_model = final_als.fit(train_df)
    log("Final model trained.")

else:
    log("Skipping grid search — training with default parameters...")
    log("  rank=20, regParam=0.1, maxIter=10")

    best_rank    = 20
    best_reg     = 0.1
    best_maxIter = 10

    als = ALS(
        rank=best_rank,
        regParam=best_reg,
        maxIter=best_maxIter,
        userCol="user_index",
        itemCol="product_index",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        implicitPrefs=False
    )
    best_model = als.fit(train_df)

# ─────────────────────────────────────────
# MODEL EVALUATION
# ─────────────────────────────────────────
section("STEP 6 — MODEL EVALUATION")

train_predictions = best_model.transform(train_df)
test_predictions  = best_model.transform(test_df)

train_rmse = round(evaluator.evaluate(train_predictions), 4)
test_rmse  = round(evaluator.evaluate(test_predictions),  4)

mae_evaluator = RegressionEvaluator(
    metricName="mae",
    labelCol="rating",
    predictionCol="prediction"
)
test_mae = round(mae_evaluator.evaluate(test_predictions), 4)

log(f"Train RMSE : {train_rmse}")
log(f"Test RMSE  : {test_rmse}")
log(f"Test MAE   : {test_mae}")

pred_stats = test_predictions.select(
    F.min("prediction").alias("min_pred"),
    F.max("prediction").alias("max_pred"),
    F.round(F.avg("prediction"), 4).alias("avg_pred"),
    F.round(F.stddev("prediction"), 4).alias("stddev_pred")
).collect()[0]

log()
log(f"Min prediction  : {pred_stats['min_pred']}")
log(f"Max prediction  : {pred_stats['max_pred']}")
log(f"Avg prediction  : {pred_stats['avg_pred']}")
log(f"Stddev          : {pred_stats['stddev_pred']}")

# ─────────────────────────────────────────
# PRECISION@10
# ─────────────────────────────────────────
section("STEP 7 — PRECISION@10 EVALUATION")

log("Computing Precision@10...")

top_k              = 10
relevant_threshold = 4.0

top10_recs = best_model.recommendForAllUsers(top_k)

recs_exploded = top10_recs.select(
    "user_index",
    F.explode("recommendations").alias("rec")
).select(
    "user_index",
    F.col("rec.product_index").alias("rec_product_index"),
    F.col("rec.rating").alias("rec_rating")
)

relevant_items = test_df.filter(
    F.col("rating") >= relevant_threshold
).select("user_index", "product_index").distinct()

hits = recs_exploded.join(
    relevant_items,
    (recs_exploded.user_index == relevant_items.user_index) &
    (recs_exploded.rec_product_index == relevant_items.product_index),
    how="inner"
).select(recs_exploded.user_index).groupBy(
    recs_exploded.user_index
).count()

total_users_with_recs = recs_exploded.select("user_index").distinct().count()
total_hits            = hits.agg(F.sum("count")).collect()[0][0] or 0
avg_precision         = round(total_hits / (total_users_with_recs * top_k), 4)

log(f"Precision@10 : {avg_precision}")
log(f"(range 0.0 to 1.0, higher is better)")

# ─────────────────────────────────────────
# GENERATE RECOMMENDATIONS
# ─────────────────────────────────────────
section("STEP 8 — GENERATING TOP 10 RECOMMENDATIONS")

log("Generating top 10 recommendations for all users...")
user_recs  = best_model.recommendForAllUsers(10)
user_count = user_recs.count()
log(f"Users with recommendations : {user_count:,}")

log()
log("Sample recommendations (2 users):")
for row in user_recs.limit(2).collect():
    log(f"  user_index : {row['user_index']}")
    recs = [(r.product_index, round(r.rating, 3)) for r in row['recommendations']]
    log(f"  top 10     : {recs}")
    log()

# ─────────────────────────────────────────
# SAVE EVERYTHING
# ─────────────────────────────────────────
section("STEP 9 — SAVING MODEL AND ARTIFACTS")

best_model.write().overwrite().save(OUTPUT_BASE + "/als_model")
log(f"ALS model saved          : {OUTPUT_BASE}/als_model")

user_recs.write.mode("overwrite").parquet(
    OUTPUT_BASE + "/user_recommendations"
)
log(f"User recommendations     : {OUTPUT_BASE}/user_recommendations")

best_model.userFactors.write.mode("overwrite").parquet(
    OUTPUT_BASE + "/user_factors"
)
best_model.itemFactors.write.mode("overwrite").parquet(
    OUTPUT_BASE + "/item_factors"
)

log(f"User factors saved       : {OUTPUT_BASE}/user_factors")
log(f"Item factors saved       : {OUTPUT_BASE}/item_factors")
log(f"User factors shape       : {best_model.userFactors.count():,} users x rank {best_rank}")
log(f"Item factors shape       : {best_model.itemFactors.count():,} items x rank {best_rank}")

summary_data = [
    ("total_interactions",  str(interaction_count)),
    ("unique_users",        str(unique_users)),
    ("unique_products",     str(unique_products)),
    ("train_count",         str(train_count)),
    ("test_count",          str(test_count)),
    ("best_rank",           str(best_rank)),
    ("best_regParam",       str(best_reg)),
    ("best_maxIter",        str(best_maxIter)),
    ("train_rmse",          str(train_rmse)),
    ("test_rmse",           str(test_rmse)),
    ("test_mae",            str(test_mae)),
    ("precision_at_10",     str(avg_precision)),
    ("users_with_recs",     str(user_count)),
    ("grid_search_used",    str(USE_GRID_SEARCH)),
    ("grid_folds",          str(GRID_FOLDS)),
]

summary_df = spark.createDataFrame(summary_data, ["metric", "value"])
summary_df.coalesce(1).write.mode("overwrite") \
    .option("header", "true").csv(OUTPUT_BASE + "/summary")

report_df = spark.createDataFrame([(l,) for l in lines], ["line"])
report_df.coalesce(1).write.mode("overwrite").text(OUTPUT_BASE + "/report")

log()
log("Summary CSV saved.")
log("Full report saved.")
log()
log("=" * 60)
log("ALS TRAINING COMPLETE")
log("=" * 60)

spark.stop()