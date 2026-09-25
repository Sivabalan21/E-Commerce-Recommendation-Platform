import sys
import time
import json
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
    print("Usage: spark-submit 08a_hpo.py <input_path> <output_path>")
    print("Example:")
    print("  spark-submit 08a_hpo.py \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/sentiment/appliances/sentiment_data \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/hpo/appliances")
    sys.exit(1)

INPUT_PATH  = sys.argv[1]
OUTPUT_BASE = sys.argv[2]

# ─────────────────────────────────────────
# HPO CONFIG
# Modify these to change search space
# ─────────────────────────────────────────
GRID_RANKS      = [10, 20, 50]
GRID_REG_PARAMS = [0.01, 0.1, 1.0]
GRID_MAX_ITERS  = [5, 10, 20]
GRID_FOLDS      = 2
SAMPLE_FRACTION = 1.0   # 1.0 = full data, 0.1 = 10% sample for speed

# ─────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("ALS_HPO") \
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

# ─────────────────────────────────────────
# HDFS HELPER FUNCTIONS
# Save and load trial results from HDFS
# ─────────────────────────────────────────
def trial_key(rank, reg, maxiter):
    return f"rank{rank}_reg{str(reg).replace('.','p')}_iter{maxiter}"

def save_trial(rank, reg, maxiter, fold_rmses, avg_rmse):
    key     = trial_key(rank, reg, maxiter)
    path    = OUTPUT_BASE + f"/trials/{key}.json"
    result  = {
        "rank":      rank,
        "regParam":  reg,
        "maxIter":   maxiter,
        "fold_rmses": fold_rmses,
        "avg_rmse":  avg_rmse,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    result_df = spark.createDataFrame(
        [(json.dumps(result),)], ["content"]
    )
    result_df.coalesce(1).write.mode("overwrite").text(path)
    log(f"  Trial saved to HDFS: {key}.json")

def load_completed_trials():
    completed = {}
    try:
        fs   = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jsc.hadoopConfiguration()
        )
        path = spark._jvm.org.apache.hadoop.fs.Path(
            OUTPUT_BASE + "/trials"
        )
        if fs.exists(path):
            for trial_dir in fs.listStatus(path):
                trial_path = str(trial_dir.getPath())
                try:
                    # Read the part file inside each trial directory
                    content = spark.read.text(trial_path).collect()
                    for row in content:
                        text = row["value"].strip()
                        if text.startswith("{"):
                            data = json.loads(text)
                            key  = trial_key(
                                data["rank"],
                                data["regParam"],
                                data["maxIter"]
                            )
                            completed[key] = data
                            log(f"  Loaded: {key} RMSE={data['avg_rmse']}")
                except Exception as e:
                    log(f"  Skipping {trial_path}: {e}")
    except Exception as e:
        log(f"  No existing trials. Starting fresh. ({e})")
    return completed

def save_best_params(best_params, best_rmse, all_results):
    result = {
        "best_rank":     best_params[0],
        "best_regParam": best_params[1],
        "best_maxIter":  best_params[2],
        "best_rmse":     best_rmse,
        "all_results":   all_results,
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S")
    }
    result_df = spark.createDataFrame(
        [(json.dumps(result, indent=2),)], ["content"]
    )
    result_df.coalesce(1).write.mode("overwrite").text(
        OUTPUT_BASE + "/best_params"
    )
    log(f"Best params saved to HDFS.")

# ─────────────────────────────────────────
# STEP 1 — LOAD AND PREPARE
# ─────────────────────────────────────────
section("STEP 1 — LOADING AND PREPARING DATA")

df = spark.read.parquet(INPUT_PATH)

interactions = df.select(
    "user_id", "parent_asin", "rating"
).dropDuplicates(["user_id", "parent_asin"])

if SAMPLE_FRACTION < 1.0:
    log(f"Sampling {SAMPLE_FRACTION*100}% of data for HPO...")
    interactions = interactions.sample(
        fraction=SAMPLE_FRACTION, seed=42
    )

interactions = interactions.repartition(200)
interactions.persist(StorageLevel.MEMORY_AND_DISK)

interaction_count = interactions.count()
log(f"Total interactions : {interaction_count:,}")
log(f"Sample fraction    : {SAMPLE_FRACTION}")

stats = interactions.agg(
    F.approx_count_distinct("user_id").alias("unique_users"),
    F.approx_count_distinct("parent_asin").alias("unique_products")
).collect()[0]

log(f"Unique users       : {stats['unique_users']:,}")
log(f"Unique products    : {stats['unique_products']:,}")

# ─────────────────────────────────────────
# STEP 2 — STRING INDEXING
# ─────────────────────────────────────────
section("STEP 2 — STRING INDEXING")

log("Fitting StringIndexers in one pipeline pass...")

user_indexer    = StringIndexer(inputCol="user_id",     outputCol="user_index")
product_indexer = StringIndexer(inputCol="parent_asin", outputCol="product_index")
pipeline        = Pipeline(stages=[user_indexer, product_indexer])
pipeline_model  = pipeline.fit(interactions)

df_indexed = pipeline_model.transform(interactions) \
    .withColumn("user_index",    F.col("user_index").cast(IntegerType())) \
    .withColumn("product_index", F.col("product_index").cast(IntegerType())) \
    .select("user_index", "product_index", "rating")

df_indexed.persist(StorageLevel.MEMORY_AND_DISK)
interactions.unpersist()

log("String indexing complete.")

# ─────────────────────────────────────────
# STEP 3 — LOAD COMPLETED TRIALS
# Skip trials already done in previous runs
# ─────────────────────────────────────────
section("STEP 3 — CHECKING COMPLETED TRIALS")

completed_trials = load_completed_trials()
log(f"Completed trials found : {len(completed_trials)}")

param_grid   = list(itertools.product(
    GRID_RANKS, GRID_REG_PARAMS, GRID_MAX_ITERS
))
total_trials = len(param_grid)
remaining    = [
    (r, rp, mi) for r, rp, mi in param_grid
    if trial_key(r, rp, mi) not in completed_trials
]

log(f"Total combinations     : {total_trials}")
log(f"Already completed      : {len(completed_trials)}")
log(f"Remaining to run       : {len(remaining)}")

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

# ─────────────────────────────────────────
# STEP 4 — RUN HPO
# Each trial saved immediately to HDFS
# ─────────────────────────────────────────
section("STEP 4 — HYPERPARAMETER SEARCH")

log(f"Search space:")
log(f"  rank     : {GRID_RANKS}")
log(f"  regParam : {GRID_REG_PARAMS}")
log(f"  maxIter  : {GRID_MAX_ITERS}")
log(f"  folds    : {GRID_FOLDS}")
log()

job_start = time.time()

for i, (rank, reg_param, max_iter) in enumerate(remaining):
    trial_start = time.time()
    done_count  = len(completed_trials) + i + 1

    log(f"[{done_count:>2}/{total_trials}] rank={rank:<4} regParam={reg_param:<6} maxIter={max_iter:<4} starting...")

    fold_rmse  = []
    folds_data = df_indexed.randomSplit(
        [1.0/GRID_FOLDS] * GRID_FOLDS, seed=42
    )

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
            nonnegative=True
        )

        model_cv  = als_trial.fit(train_fold)
        preds_cv  = model_cv.transform(val_fold)
        rmse_fold = round(evaluator.evaluate(preds_cv), 4)
        fold_rmse.append(rmse_fold)
        log(f"  fold {fold+1}/{GRID_FOLDS} RMSE={rmse_fold}")

    avg_rmse      = round(sum(fold_rmse) / len(fold_rmse), 4)
    elapsed       = round(time.time() - trial_start, 1)
    total_elapsed = round((time.time() - job_start) / 60, 1)

    log(f"  avg RMSE={avg_rmse}  trial={elapsed}s  total={total_elapsed}min")

    # Save immediately to HDFS
    save_trial(rank, reg_param, max_iter, fold_rmse, avg_rmse)

    completed_trials[trial_key(rank, reg_param, max_iter)] = {
        "rank":     rank,
        "regParam": reg_param,
        "maxIter":  max_iter,
        "avg_rmse": avg_rmse
    }
    log()

# ─────────────────────────────────────────
# STEP 5 — FIND BEST PARAMS FROM ALL TRIALS
# ─────────────────────────────────────────
section("STEP 5 — RESULTS SUMMARY")

all_results   = []
best_rmse     = float("inf")
best_params   = None

for key, data in completed_trials.items():
    all_results.append({
        "rank":     data["rank"],
        "regParam": data["regParam"],
        "maxIter":  data["maxIter"],
        "avg_rmse": data["avg_rmse"]
    })
    if data["avg_rmse"] < best_rmse:
        best_rmse   = data["avg_rmse"]
        best_params = (data["rank"], data["regParam"], data["maxIter"])

all_results_sorted = sorted(all_results, key=lambda x: x["avg_rmse"])

log("All trial results (sorted by RMSE):")
log(f"{'#':<5} {'rank':<8} {'regParam':<12} {'maxIter':<10} {'avg_RMSE':>10}")
log("-" * 50)
for idx, r in enumerate(all_results_sorted):
    marker = " <-- BEST" if (r["rank"], r["regParam"], r["maxIter"]) == best_params else ""
    log(f"{idx+1:<5} {r['rank']:<8} {r['regParam']:<12} {r['maxIter']:<10} {r['avg_rmse']:>10}{marker}")

log()
log(f"Best rank     : {best_params[0]}")
log(f"Best regParam : {best_params[1]}")
log(f"Best maxIter  : {best_params[2]}")
log(f"Best avg RMSE : {best_rmse}")

# Save best params
save_best_params(best_params, best_rmse, all_results_sorted)

# ─────────────────────────────────────────
# STEP 6 — SAVE REPORT
# ─────────────────────────────────────────
section("STEP 6 — SAVING REPORT")

report_df = spark.createDataFrame([(l,) for l in lines], ["line"])
report_df.coalesce(1).write.mode("overwrite").text(
    OUTPUT_BASE + "/report"
)
log("Report saved.")
log()
log("=" * 60)
log("HPO COMPLETE")
log("=" * 60)
log()
log("Next step: run 08b_als_final_model.py with best params above.")

spark.stop()