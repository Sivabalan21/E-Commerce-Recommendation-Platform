import sys
import json
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexerModel

# ─────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────
if len(sys.argv) != 3:
    print("Usage: spark-submit 11_export_artifacts.py "
          "<output_local_dir> <output_hdfs_dir>")
    print("Example:")
    print("  spark-submit 11_export_artifacts.py "
          "~/ecommerce/artifacts "
          "hdfs:///user/ss21176_nyu_edu/ecommerce/artifacts/appliances")
    sys.exit(1)

OUTPUT_LOCAL = sys.argv[1]
OUTPUT_HDFS  = sys.argv[2]

# ─────────────────────────────────────────
# FIXED PATHS
# ─────────────────────────────────────────
USER_FACTORS_PATH    = "hdfs:///user/ss21176_nyu_edu/ecommerce/models/appliances/user_factors"
ITEM_FACTORS_PATH    = "hdfs:///user/ss21176_nyu_edu/ecommerce/models/appliances/item_factors"
USER_INDEXER_PATH    = "hdfs:///user/ss21176_nyu_edu/ecommerce/models/appliances/user_indexer"
PRODUCT_INDEXER_PATH = "hdfs:///user/ss21176_nyu_edu/ecommerce/models/appliances/product_indexer"
PAGERANK_PATH        = "hdfs:///user/ss21176_nyu_edu/ecommerce/graph/appliances/pagerank_scores"
METADATA_PATH        = "hdfs:///user/ss21176_nyu_edu/ecommerce/cleaning/appliances/metadata/cleaned_metadata"
SENTIMENT_PATH       = "hdfs:///user/ss21176_nyu_edu/ecommerce/sentiment/appliances/sentiment_data"

# ─────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("Export_Artifacts") \
    .master("local[*]") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.driver.memory",        "8g") \
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

import os
log(f"Output local dir : {OUTPUT_LOCAL}")
log(f"Output HDFS dir  : {OUTPUT_HDFS}")

# ─────────────────────────────────────────
# STEP 1 — EXPORT USER FACTORS
# Load Parquet, convert to numpy float32
# ─────────────────────────────────────────
section("STEP 1 — EXPORTING USER FACTORS")

log("Loading user_factors from HDFS...")
user_factors_df = spark.read.parquet(USER_FACTORS_PATH) \
    .orderBy("id")

user_count = user_factors_df.count()
log(f"Users : {user_count:,}")

log("Converting to numpy array...")
user_factors_list = user_factors_df.select("features").toPandas()["features"].tolist()
user_factors_np   = np.array(user_factors_list, dtype=np.float32)

log(f"user_factors shape : {user_factors_np.shape}")

user_factors_path = f"{OUTPUT_LOCAL}/user_factors.npy"
np.save(user_factors_path, user_factors_np)
log(f"Saved : {user_factors_path}")

del user_factors_list
del user_factors_np

# ─────────────────────────────────────────
# STEP 2 — EXPORT ITEM FACTORS
# ─────────────────────────────────────────
section("STEP 2 — EXPORTING ITEM FACTORS")

log("Loading item_factors from HDFS...")
item_factors_df = spark.read.parquet(ITEM_FACTORS_PATH) \
    .orderBy("id")

item_count = item_factors_df.count()
log(f"Products : {item_count:,}")

log("Converting to numpy array...")
item_factors_list = item_factors_df.select("features").toPandas()["features"].tolist()
item_factors_np   = np.array(item_factors_list, dtype=np.float32)

log(f"item_factors shape : {item_factors_np.shape}")

item_factors_path = f"{OUTPUT_LOCAL}/item_factors.npy"
np.save(item_factors_path, item_factors_np)
log(f"Saved : {item_factors_path}")

del item_factors_list
del item_factors_np

# ─────────────────────────────────────────
# STEP 3 — EXPORT USER ID MAPPINGS
# user_id string → row index in user_factors
# ─────────────────────────────────────────
section("STEP 3 — EXTRACTING USER ID MAPPINGS")

log("Reading user_indexer data Parquet...")
user_indexer_parquet = sorted(
    Path(f"{ARTIFACTS_DIR}/user_indexer/data").glob("*.parquet")
)[0]

t          = pq.read_table(str(user_indexer_parquet))
df         = t.to_pandas()
user_labels = df["labelsArray"][0][0]

log(f"Total user labels : {len(user_labels):,}")

user_id_to_index = {uid: idx for idx, uid in enumerate(user_labels)}

with open(f"{OUTPUT_DIR}/user_id_to_index.json", "w") as f:
    json.dump(user_id_to_index, f)

log(f"Saved: user_id_to_index.json")
log(f"Sample: {list(user_id_to_index.items())[:2]}")

# ─────────────────────────────────────────
# STEP 4 — EXPORT PRODUCT INDEX MAPPINGS
# column index → parent_asin string
# ─────────────────────────────────────────
section("STEP 4 — EXTRACTING PRODUCT INDEX MAPPINGS")

log("Reading product_indexer data Parquet...")
product_indexer_parquet = sorted(
    Path(f"{ARTIFACTS_DIR}/product_indexer/data").glob("*.parquet")
)[0]

t              = pq.read_table(str(product_indexer_parquet))
df             = t.to_pandas()
product_labels = df["labelsArray"][0][0]

log(f"Total product labels : {len(product_labels):,}")

index_to_asin = {str(idx): asin for idx, asin in enumerate(product_labels)}
asin_to_index = {asin: idx for idx, asin in enumerate(product_labels)}

with open(f"{OUTPUT_DIR}/index_to_asin.json", "w") as f:
    json.dump(index_to_asin, f)

with open(f"{OUTPUT_DIR}/asin_to_index.json", "w") as f:
    json.dump(asin_to_index, f)

log("Saved: index_to_asin.json")
log("Saved: asin_to_index.json")
log(f"Sample: {list(index_to_asin.items())[:2]}")

# ─────────────────────────────────────────
# STEP 5 — EXPORT PAGERANK CSV
# ─────────────────────────────────────────
section("STEP 5 — EXPORTING PAGERANK SCORES")

log("Loading PageRank scores...")
df_pr = spark.read.parquet(PAGERANK_PATH).select(
    "parent_asin",
    "pagerank_score",
    "pagerank_normalized"
)

pr_count = df_pr.count()
log(f"Products with PageRank : {pr_count:,}")

pr_path = f"{OUTPUT_LOCAL}/pagerank.csv"
df_pr.toPandas().to_csv(pr_path, index=False)
log(f"Saved : {pr_path}")

# ─────────────────────────────────────────
# STEP 6 — EXPORT METADATA CSV
# Join metadata with sentiment labels
# ─────────────────────────────────────────
section("STEP 6 — EXPORTING METADATA")

log("Loading metadata and sentiment data...")

df_meta = spark.read.parquet(METADATA_PATH).select(
    "parent_asin", "title", "brand",
    "subcategory", "main_category",
    "price", "image_url", "average_rating"
)

df_sentiment_labels = spark.read.parquet(SENTIMENT_PATH).select(
    "parent_asin", "sentiment_label"
).groupBy("parent_asin").agg(
    F.first("sentiment_label").alias("sentiment_label")
)

df_meta_enriched = df_meta.join(
    df_sentiment_labels,
    on="parent_asin",
    how="left"
)

meta_count = df_meta_enriched.count()
log(f"Products in metadata : {meta_count:,}")

meta_path = f"{OUTPUT_LOCAL}/metadata.csv"
df_meta_enriched.toPandas().to_csv(meta_path, index=False)
log(f"Saved : {meta_path}")

# ─────────────────────────────────────────
# STEP 7 — VERIFY ALL FILES
# ─────────────────────────────────────────
section("STEP 7 — VERIFYING EXPORTED FILES")

import os
files = [
    "user_factors.npy",
    "item_factors.npy",
    "user_id_to_index.json",
    "index_to_asin.json",
    "asin_to_index.json",
    "pagerank.csv",
    "metadata.csv"
]

log(f"{'File':<30} {'Size':>12}")
log("-" * 44)
total_size = 0
for fname in files:
    fpath = f"{OUTPUT_LOCAL}/{fname}"
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        total_size += size
        log(f"{fname:<30} {size/1024/1024:>10.1f}MB")
    else:
        log(f"{fname:<30} MISSING")

log(f"{'Total':<30} {total_size/1024/1024:>10.1f}MB")

# ─────────────────────────────────────────
# STEP 8 — COPY TO HDFS FOR BACKUP
# ─────────────────────────────────────────
section("STEP 8 — BACKING UP TO HDFS")

import subprocess
for fname in files:
    fpath = f"{OUTPUT_LOCAL}/{fname}"
    if os.path.exists(fpath):
        cmd = f"hadoop fs -put -f {fpath} {OUTPUT_HDFS}/{fname}"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        if result.returncode == 0:
            log(f"Backed up to HDFS: {fname}")
        else:
            log(f"Failed to backup: {fname} — {result.stderr.decode()}")

# ─────────────────────────────────────────
# SAVE REPORT
# ─────────────────────────────────────────
section("SAVING REPORT")

report_df = spark.createDataFrame([(l,) for l in lines], ["line"])
report_df.coalesce(1).write.mode("overwrite").text(
    OUTPUT_HDFS + "/report"
)

log()
log("=" * 60)
log("EXPORT ARTIFACTS COMPLETE")
log("=" * 60)
log()
log("Files ready for RunPod upload:")
for fname in files:
    log(f"  {OUTPUT_LOCAL}/{fname}")

spark.stop()