import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

# ─────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────
if len(sys.argv) != 3:
    print("Usage: spark-submit 02_profiling_metadata.py <input_path> <output_path>")
    print("Example:")
    print("  spark-submit 02_profiling_metadata.py \\")
    print("    hdfs:///user/mc10322_nyu_edu/ecommerce/raw/appliances_metadata.json.gz \\")
    print("    hdfs:///user/mc10322_nyu_edu/ecommerce/profiling/appliances/metadata")
    sys.exit(1)

METADATA_PATH = sys.argv[1]
OUTPUT_BASE   = sys.argv[2]
OUTPUT_CSV    = OUTPUT_BASE + "/summary"
OUTPUT_TXT    = OUTPUT_BASE + "/report"

# ─────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("Metadata_Profiling") \
    .config("spark.sql.adaptive.enabled", "true") \
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
# LOAD AND CACHE
# ─────────────────────────────────────────
# ─────────────────────────────────────────
# LOAD AND CACHE
# ─────────────────────────────────────────
section("STEP 1 — LOADING METADATA DATASET")

from pyspark.sql.types import StructType, StructField, StringType, \
    DoubleType, LongType, ArrayType, MapType

schema = StructType([
    StructField("parent_asin",    StringType(),  True),
    StructField("title",          StringType(),  True),
    StructField("main_category",  StringType(),  True),
    StructField("store",          StringType(),  True),
    StructField("price",          DoubleType(),  True),
    StructField("average_rating", DoubleType(),  True),
    StructField("rating_number",  LongType(),    True),
    StructField("features",       ArrayType(StringType()), True),
    StructField("description",    ArrayType(StringType()), True),
    StructField("categories",     ArrayType(StringType()), True),
    StructField("bought_together", StringType(), True),
    StructField("images", ArrayType(
        StructType([
            StructField("thumb",   StringType(), True),
            StructField("large",   StringType(), True),
            StructField("variant", StringType(), True),
            StructField("hi_res",  StringType(), True),
        ])
    ), True),
])

df = spark.read.schema(schema).json(METADATA_PATH)
df.cache()

total_rows = df.count()
log(f"Total rows     : {total_rows:,}")
log(f"Total columns  : {len(df.columns)}")
log(f"Columns        : {df.columns}")

# ─────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────
section("STEP 2 — SCHEMA")

for field in df.schema.fields:
    log(f"  {field.name:<25} {str(field.dataType):<30} nullable={field.nullable}")

log()
log("Sample records (2 rows):")
sample_rows = df.select(
    "parent_asin", "title", "main_category", "store",
    "price", "average_rating", "rating_number"
).limit(2).collect()

for row in sample_rows:
    log(f"  parent_asin    : {row['parent_asin']}")
    log(f"  title          : {str(row['title'])[:80]}")
    log(f"  main_category  : {row['main_category']}")
    log(f"  store          : {row['store']}")
    log(f"  price          : {row['price']}")
    log(f"  average_rating : {row['average_rating']}")
    log(f"  rating_number  : {row['rating_number']}")
    log()

# ─────────────────────────────────────────
# NULL ANALYSIS — single pass
# ─────────────────────────────────────────
section("STEP 3 — NULL AND MISSING VALUE ANALYSIS")

cols_to_check = [
    "parent_asin", "title", "main_category", "store",
    "price", "average_rating", "rating_number",
    "features", "description", "categories",
    "images", "bought_together"
]

null_exprs = [
    F.sum(F.when(
        F.col(c).isNull() | (F.col(c).cast("string") == "") |
        (F.col(c).cast("string") == "null"),
        1).otherwise(0)).alias(c)
    for c in cols_to_check if c in df.columns
]

null_counts = df.select(null_exprs).collect()[0]

log(f"\n{'Column':<25} {'Null Count':>12} {'Null %':>10} {'Non-Null':>12}")
log("-" * 62)
for c in cols_to_check:
    if c in df.columns:
        nc  = null_counts[c]
        pct = round(100 * nc / total_rows, 4)
        log(f"{c:<25} {nc:>12,} {pct:>9}% {(total_rows-nc):>12,}")

# ─────────────────────────────────────────
# PRICE ANALYSIS — single pass
# ─────────────────────────────────────────
section("STEP 4 — PRICE FIELD ANALYSIS")

price_stats = df.select(
    F.sum(F.when(F.col("price").isNull(), 1)
            .otherwise(0)).alias("null_price"),
    F.sum(F.when(F.col("price").isNotNull(), 1)
            .otherwise(0)).alias("has_price"),
    F.min("price").alias("min_price"),
    F.max("price").alias("max_price"),
    F.round(F.avg("price"), 2).alias("avg_price"),
    F.round(F.stddev("price"), 2).alias("stddev_price"),
    F.sum(F.when(F.col("price") <= 0, 1)
            .otherwise(0)).alias("zero_or_negative"),
    F.sum(F.when(F.col("price") > 1000, 1)
            .otherwise(0)).alias("over_1000"),
).collect()[0]

null_pct  = round(100 * price_stats["null_price"]  / total_rows, 2)
has_pct   = round(100 * price_stats["has_price"]   / total_rows, 2)

log(f"Products with price      : {price_stats['has_price']:>10,}  ({has_pct}%)")
log(f"Products without price   : {price_stats['null_price']:>10,}  ({null_pct}%)")
log(f"Min price                : {price_stats['min_price']}")
log(f"Max price                : {price_stats['max_price']}")
log(f"Avg price                : {price_stats['avg_price']}")
log(f"Stddev price             : {price_stats['stddev_price']}")
log(f"Zero or negative price   : {price_stats['zero_or_negative']:>10,}")
log(f"Price over $1000         : {price_stats['over_1000']:>10,}")

log()
log("Price range distribution:")
log(f"{'Range':<20} {'Count':>12} {'Percentage':>12}")
log("-" * 46)

price_buckets = df.filter(F.col("price").isNotNull()).select(
    F.when(F.col("price") <= 0,     "zero or negative")
     .when(F.col("price") < 10,     "$0.01 - $9.99")
     .when(F.col("price") < 50,     "$10 - $49.99")
     .when(F.col("price") < 100,    "$50 - $99.99")
     .when(F.col("price") < 500,    "$100 - $499.99")
     .when(F.col("price") < 1000,   "$500 - $999.99")
     .otherwise("$1000+")
     .alias("range")
).groupBy("range").count().orderBy("range").collect()

for row in price_buckets:
    pct = round(100 * row["count"] / total_rows, 2)
    log(f"{row['range']:<20} {row['count']:>12,} {pct:>11}%")

# ─────────────────────────────────────────
# AVERAGE RATING AND RATING NUMBER
# ─────────────────────────────────────────
section("STEP 5 — AVERAGE RATING AND RATING NUMBER ANALYSIS")

rating_stats = df.select(
    F.sum(F.when(F.col("average_rating").isNull(), 1)
            .otherwise(0)).alias("null_avg_rating"),
    F.min("average_rating").alias("min_avg_rating"),
    F.max("average_rating").alias("max_avg_rating"),
    F.round(F.avg("average_rating"), 4).alias("mean_avg_rating"),
    F.sum(F.when(F.col("average_rating").isNull(), 1)
            .otherwise(0)).alias("null_rating_number"),
    F.min("rating_number").alias("min_rating_number"),
    F.max("rating_number").alias("max_rating_number"),
    F.round(F.avg("rating_number"), 2).alias("mean_rating_number"),
    F.sum(F.when(F.col("rating_number") == 0, 1)
            .otherwise(0)).alias("zero_rating_number"),
    F.sum(F.when(F.col("rating_number") < 5, 1)
            .otherwise(0)).alias("under_5_ratings"),
    F.sum(F.when(F.col("rating_number") > 1000, 1)
            .otherwise(0)).alias("over_1000_ratings"),
).collect()[0]

log(f"Null average_rating      : {rating_stats['null_avg_rating']:>10,}")
log(f"Min average_rating       : {rating_stats['min_avg_rating']}")
log(f"Max average_rating       : {rating_stats['max_avg_rating']}")
log(f"Mean average_rating      : {rating_stats['mean_avg_rating']}")
log()
log(f"Null rating_number       : {rating_stats['null_rating_number']:>10,}")
log(f"Min rating_number        : {rating_stats['min_rating_number']}")
log(f"Max rating_number        : {rating_stats['max_rating_number']:,}")
log(f"Mean rating_number       : {rating_stats['mean_rating_number']}")
log(f"Zero rating_number       : {rating_stats['zero_rating_number']:>10,}")
log(f"Under 5 ratings          : {rating_stats['under_5_ratings']:>10,}")
log(f"Over 1000 ratings        : {rating_stats['over_1000_ratings']:>10,}")

# ─────────────────────────────────────────
# CATEGORIES ANALYSIS
# ─────────────────────────────────────────
section("STEP 6 — CATEGORIES FIELD ANALYSIS")

cat_stats = df.select(
    F.sum(F.when(F.col("categories").isNull() |
                 (F.size(F.col("categories")) == 0), 1)
            .otherwise(0)).alias("null_or_empty_categories"),
    F.min(F.size(F.col("categories"))).alias("min_depth"),
    F.max(F.size(F.col("categories"))).alias("max_depth"),
    F.round(F.avg(F.size(F.col("categories"))), 2).alias("avg_depth"),
).collect()[0]

log(f"Null or empty categories : {cat_stats['null_or_empty_categories']:>10,}")
log(f"Min category depth       : {cat_stats['min_depth']}")
log(f"Max category depth       : {cat_stats['max_depth']}")
log(f"Avg category depth       : {cat_stats['avg_depth']}")

log()
log("Top 15 main categories by product count:")
log(f"{'main_category':<45} {'count':>10} {'percentage':>12}")
log("-" * 68)

top_cats = df.groupBy("main_category").count() \
             .orderBy(F.col("count").desc()) \
             .limit(15).collect()
for row in top_cats:
    pct = round(100 * row["count"] / total_rows, 2)
    cat = str(row["main_category"])[:44]
    log(f"{cat:<45} {row['count']:>10,} {pct:>11}%")

log()
log("Top 15 leaf subcategories (last element of categories array):")
log(f"{'subcategory':<45} {'count':>10}")
log("-" * 56)

leaf_cats = df.filter(
    F.col("categories").isNotNull() & (F.size(F.col("categories")) > 0)
).withColumn(
    "leaf_cat", F.element_at(F.col("categories"), -1)
).groupBy("leaf_cat").count() \
 .orderBy(F.col("count").desc()) \
 .limit(15).collect()

for row in leaf_cats:
    cat = str(row["leaf_cat"])[:44]
    log(f"{cat:<45} {row['count']:>10,}")

# ─────────────────────────────────────────
# IMAGES ANALYSIS
# ─────────────────────────────────────────
section("STEP 7 — IMAGES FIELD ANALYSIS")

img_stats = df.select(
    F.sum(F.when(F.col("images").isNull() |
                 (F.size(F.col("images")) == 0), 1)
            .otherwise(0)).alias("no_images"),
    F.sum(F.when(F.col("images").isNotNull() &
                 (F.size(F.col("images")) > 0), 1)
            .otherwise(0)).alias("has_images"),
    F.round(F.avg(F.size(F.col("images"))), 2).alias("avg_image_count"),
    F.max(F.size(F.col("images"))).alias("max_image_count"),
).collect()[0]

log(f"Products with images     : {img_stats['has_images']:>10,}  ({round(100*img_stats['has_images']/total_rows,2)}%)")
log(f"Products without images  : {img_stats['no_images']:>10,}  ({round(100*img_stats['no_images']/total_rows,2)}%)")
log(f"Avg images per product   : {img_stats['avg_image_count']}")
log(f"Max images per product   : {img_stats['max_image_count']}")

log()
log("Checking hi_res image URL availability:")
hi_res_stats = df.filter(
    F.col("images").isNotNull() & (F.size(F.col("images")) > 0)
).select(
    F.sum(F.when(
        F.col("images")[0]["hi_res"].isNotNull() &
        (F.col("images")[0]["hi_res"] != ""),
        1).otherwise(0)).alias("has_hi_res"),
    F.sum(F.when(
        F.col("images")[0]["hi_res"].isNull() |
        (F.col("images")[0]["hi_res"] == ""),
        1).otherwise(0)).alias("no_hi_res"),
).collect()[0]

log(f"Products with hi_res URL : {hi_res_stats['has_hi_res']:>10,}")
log(f"Products missing hi_res  : {hi_res_stats['no_hi_res']:>10,}")

# ─────────────────────────────────────────
# STORE / BRAND ANALYSIS
# ─────────────────────────────────────────
section("STEP 8 — STORE (BRAND) FIELD ANALYSIS")

store_stats = df.select(
    F.sum(F.when(F.col("store").isNull() |
                 (F.col("store") == ""), 1)
            .otherwise(0)).alias("null_store"),
    F.approx_count_distinct("store").alias("unique_stores"),
).collect()[0]

log(f"Null or empty store      : {store_stats['null_store']:>10,}  ({round(100*store_stats['null_store']/total_rows,2)}%)")
log(f"Unique stores (brands)   : {store_stats['unique_stores']:>10,}")

log()
log("Top 15 stores by product count:")
log(f"{'store':<35} {'count':>10} {'percentage':>12}")
log("-" * 58)

top_stores = df.filter(
    F.col("store").isNotNull() & (F.col("store") != "")
).groupBy("store").count() \
 .orderBy(F.col("count").desc()) \
 .limit(15).collect()

for row in top_stores:
    store = str(row["store"])[:34]
    pct   = round(100 * row["count"] / total_rows, 2)
    log(f"{store:<35} {row['count']:>10,} {pct:>11}%")

# ─────────────────────────────────────────
# FEATURES AND DESCRIPTION
# ─────────────────────────────────────────
section("STEP 9 — FEATURES AND DESCRIPTION ANALYSIS")

feat_stats = df.select(
    F.sum(F.when(F.col("features").isNull() |
                 (F.size(F.col("features")) == 0), 1)
            .otherwise(0)).alias("no_features"),
    F.sum(F.when(F.col("features").isNotNull() &
                 (F.size(F.col("features")) > 0), 1)
            .otherwise(0)).alias("has_features"),
    F.round(F.avg(F.size(F.col("features"))), 2).alias("avg_features"),
    F.max(F.size(F.col("features"))).alias("max_features"),
    F.sum(F.when(F.col("description").isNull() |
                 (F.size(F.col("description")) == 0), 1)
            .otherwise(0)).alias("no_description"),
    F.sum(F.when(F.col("description").isNotNull() &
                 (F.size(F.col("description")) > 0), 1)
            .otherwise(0)).alias("has_description"),
).collect()[0]

log(f"Products with features    : {feat_stats['has_features']:>10,}  ({round(100*feat_stats['has_features']/total_rows,2)}%)")
log(f"Products without features : {feat_stats['no_features']:>10,}  ({round(100*feat_stats['no_features']/total_rows,2)}%)")
log(f"Avg features per product  : {feat_stats['avg_features']}")
log(f"Max features per product  : {feat_stats['max_features']}")
log()
log(f"Products with description : {feat_stats['has_description']:>10,}  ({round(100*feat_stats['has_description']/total_rows,2)}%)")
log(f"Products without desc     : {feat_stats['no_description']:>10,}  ({round(100*feat_stats['no_description']/total_rows,2)}%)")

# ─────────────────────────────────────────
# BOUGHT TOGETHER
# ─────────────────────────────────────────
section("STEP 10 — BOUGHT TOGETHER ANALYSIS")

bought_stats = df.select(
    F.sum(F.when(F.col("bought_together").isNull(), 1)
            .otherwise(0)).alias("null_bought"),
    F.sum(F.when(F.col("bought_together").isNotNull(), 1)
            .otherwise(0)).alias("has_bought"),
).collect()[0]

log(f"Null bought_together     : {bought_stats['null_bought']:>10,}  ({round(100*bought_stats['null_bought']/total_rows,2)}%)")
log(f"Has bought_together      : {bought_stats['has_bought']:>10,}  ({round(100*bought_stats['has_bought']/total_rows,2)}%)")
log()
log("NOTE: bought_together is null for all products in this dataset.")
log("Co-purchase graph will be built from review co-occurrence instead.")

# ─────────────────────────────────────────
# DUPLICATE DETECTION
# ─────────────────────────────────────────
section("STEP 12 — DUPLICATE DETECTION")

exact_dupes   = total_rows - df.dropDuplicates(["parent_asin"]).count()
title_dupes   = df.groupBy("title").count() \
                  .filter(F.col("count") > 1).count()

log(f"Duplicate parent_asin     : {exact_dupes:>10,}")
log(f"Duplicate titles          : {title_dupes:>10,}")

log()
log("Top 5 most duplicated titles:")
log(f"{'title':<55} {'count':>6}")
log("-" * 62)
top_title_dupes = df.groupBy("title").count() \
                    .filter(F.col("count") > 1) \
                    .orderBy(F.col("count").desc()) \
                    .limit(5).collect()
for row in top_title_dupes:
    title = str(row["title"])[:54]
    log(f"{title:<55} {row['count']:>6}")

# ─────────────────────────────────────────
# UNIQUE COUNTS
# ─────────────────────────────────────────
section("STEP 13 — UNIQUE COUNTS")

unique_stats = df.select(
    F.approx_count_distinct("parent_asin").alias("unique_parent_asins"),
    F.approx_count_distinct("main_category").alias("unique_main_categories"),
    F.approx_count_distinct("store").alias("unique_stores"),
).collect()[0]

log(f"Unique parent ASINs       : {unique_stats['unique_parent_asins']:>10,}")
log(f"Unique main categories    : {unique_stats['unique_main_categories']:>10,}")
log(f"Unique stores             : {unique_stats['unique_stores']:>10,}")

# ─────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────
section("STEP 14 — SAVING OUTPUTS")

summary_data = [
    ("total_rows",                      str(total_rows)),
    ("total_columns",                   str(len(df.columns))),
    ("unique_parent_asins",             str(unique_stats["unique_parent_asins"])),
    ("unique_main_categories",          str(unique_stats["unique_main_categories"])),
    ("unique_stores",                   str(unique_stats["unique_stores"])),
    ("null_price_pct",                  str(round(100*price_stats["null_price"]/total_rows,2))),
    ("avg_price",                       str(price_stats["avg_price"])),
    ("min_price",                       str(price_stats["min_price"])),
    ("max_price",                       str(price_stats["max_price"])),
    ("mean_avg_rating",                 str(rating_stats["mean_avg_rating"])),
    ("mean_rating_number",              str(rating_stats["mean_rating_number"])),
    ("pct_has_images",                  str(round(100*img_stats["has_images"]/total_rows,2))),
    ("pct_has_hi_res_image",            str(round(100*hi_res_stats["has_hi_res"]/total_rows,2))),
    ("pct_has_features",                str(round(100*feat_stats["has_features"]/total_rows,2))),
    ("pct_has_description",             str(round(100*feat_stats["has_description"]/total_rows,2))),
    ("pct_null_bought_together",        str(round(100*bought_stats["null_bought"]/total_rows,2))),
    ("duplicate_parent_asins",          str(exact_dupes)),
    ("duplicate_titles",                str(title_dupes)),
    ("pct_null_store",                  str(round(100*store_stats["null_store"]/total_rows,2))),
    ("avg_category_depth",              str(cat_stats["avg_depth"])),
    ("avg_images_per_product",          str(img_stats["avg_image_count"])),
]

summary_df = spark.createDataFrame(summary_data, ["metric", "value"])
summary_df.coalesce(1).write.mode("overwrite") \
    .option("header", "true").csv(OUTPUT_CSV)

report_df = spark.createDataFrame([(l,) for l in lines], ["line"])
report_df.coalesce(1).write.mode("overwrite").text(OUTPUT_TXT)

log(f"Summary CSV saved.")
log(f"Full report saved.")
log()
log("=" * 60)
log("PROFILING COMPLETE")
log("=" * 60)

spark.stop()