import os
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

# ─────────────────────────────────────────
# PATHS
# Update ARTIFACTS_DIR if needed
# ─────────────────────────────────────────
import sys
ARTIFACTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "/workspace/artifacts"
OUTPUT_DIR    = sys.argv[2] if len(sys.argv) > 2 else "/workspace/artifacts"

def log(msg=""):
    print(msg)

def section(title):
    log()
    log("=" * 60)
    log(title)
    log("=" * 60)

# ─────────────────────────────────────────
# STEP 1 — CONVERT USER FACTORS
# Parquet folder → numpy float32 array
# ─────────────────────────────────────────
section("STEP 1 — CONVERTING USER FACTORS")

log("Reading user_factors Parquet files...")
user_factors_dir = f"{ARTIFACTS_DIR}/user_factors"
parquet_files    = sorted(Path(user_factors_dir).glob("part-*"))
log(f"Found {len(parquet_files)} parquet files")

dfs = []
for i, f in enumerate(parquet_files):
    df = pq.read_table(str(f)).to_pandas()
    dfs.append(df)
    log(f"  Read {f.name}: {len(df):,} rows")

user_factors_df = pd.concat(dfs, ignore_index=True)
user_factors_df = user_factors_df.sort_values("id").reset_index(drop=True)

log(f"Total users : {len(user_factors_df):,}")

user_factors_np = np.array(
    user_factors_df["features"].tolist(),
    dtype=np.float32
)

log(f"user_factors shape : {user_factors_np.shape}")

np.save(f"{OUTPUT_DIR}/user_factors.npy", user_factors_np)
log(f"Saved: user_factors.npy ({user_factors_np.nbytes/1024/1024:.1f}MB)")

del dfs, user_factors_df, user_factors_np

# ─────────────────────────────────────────
# STEP 2 — CONVERT ITEM FACTORS
# ─────────────────────────────────────────
section("STEP 2 — CONVERTING ITEM FACTORS")

log("Reading item_factors Parquet files...")
item_factors_dir = f"{ARTIFACTS_DIR}/item_factors"
parquet_files    = sorted(Path(item_factors_dir).glob("part-*"))
log(f"Found {len(parquet_files)} parquet files")

dfs = []
for f in parquet_files:
    df = pq.read_table(str(f)).to_pandas()
    dfs.append(df)

item_factors_df = pd.concat(dfs, ignore_index=True)
item_factors_df = item_factors_df.sort_values("id").reset_index(drop=True)

log(f"Total products : {len(item_factors_df):,}")

item_factors_np = np.array(
    item_factors_df["features"].tolist(),
    dtype=np.float32
)

log(f"item_factors shape : {item_factors_np.shape}")

np.save(f"{OUTPUT_DIR}/item_factors.npy", item_factors_np)
log(f"Saved: item_factors.npy ({item_factors_np.nbytes/1024/1024:.1f}MB)")

del dfs, item_factors_df, item_factors_np

# ─────────────────────────────────────────
# STEP 3 — EXTRACT USER ID MAPPINGS
# From StringIndexer metadata folder
# ─────────────────────────────────────────
section("STEP 3 — EXTRACTING USER ID MAPPINGS")

log("Reading user_indexer data Parquet...")
user_indexer_parquet = sorted(
    Path(f"{ARTIFACTS_DIR}/user_indexer/data").glob("part-*")
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
# STEP 4 — EXTRACT PRODUCT INDEX MAPPINGS
# ─────────────────────────────────────────
section("STEP 4 — EXTRACTING PRODUCT INDEX MAPPINGS")

log("Reading product_indexer data Parquet...")
product_indexer_parquet = sorted(
    Path(f"{ARTIFACTS_DIR}/product_indexer/data").glob("part-*")
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
# STEP 5 — CONVERT PAGERANK SCORES
# ─────────────────────────────────────────
section("STEP 5 — CONVERTING PAGERANK SCORES")

log("Reading pagerank_scores Parquet files...")
pagerank_dir  = f"{ARTIFACTS_DIR}/pagerank_scores"
parquet_files = sorted(Path(pagerank_dir).glob("part-*"))

dfs = []
for f in parquet_files:
    df = pq.read_table(str(f)).to_pandas()
    dfs.append(df)

pagerank_df = pd.concat(dfs, ignore_index=True)
log(f"Total products with PageRank : {len(pagerank_df):,}")

pagerank_df.to_csv(f"{OUTPUT_DIR}/pagerank.csv", index=False)
log("Saved: pagerank.csv")

# ─────────────────────────────────────────
# STEP 6 — CONVERT METADATA
# ─────────────────────────────────────────
section("STEP 6 — CONVERTING METADATA")

log("Reading cleaned_metadata Parquet files...")
metadata_dir  = f"{ARTIFACTS_DIR}/cleaned_metadata"
parquet_files = sorted(Path(metadata_dir).glob("part-*"))

dfs = []
for f in parquet_files:
    df = pq.read_table(str(f)).to_pandas()
    dfs.append(df)

metadata_df = pd.concat(dfs, ignore_index=True)
log(f"Total products in metadata : {len(metadata_df):,}")

metadata_df.to_csv(f"{OUTPUT_DIR}/metadata.csv", index=False)
log("Saved: metadata.csv")

# ─────────────────────────────────────────
# STEP 7 — VERIFY ALL FILES
# ─────────────────────────────────────────
section("STEP 7 — VERIFYING ALL FILES")

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
    fpath = f"{OUTPUT_DIR}/{fname}"
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        total_size += size
        log(f"{fname:<30} {size/1024/1024:>10.1f}MB")
    else:
        log(f"{fname:<30} MISSING")

log(f"{'TOTAL':<30} {total_size/1024/1024:>10.1f}MB")
log()
log("=" * 60)
log("CONVERSION COMPLETE — Ready for Ray Serve")
log("=" * 60)