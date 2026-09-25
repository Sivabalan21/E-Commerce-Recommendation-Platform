import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, \
    DoubleType, LongType, ArrayType

# ─────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────
if len(sys.argv) != 3:
    print("Usage: spark-submit 04_cleaning_metadata.py <input_path> <output_path>")
    print("Example:")
    print("  spark-submit 04_cleaning_metadata.py \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/raw/appliances_metadata.json.gz \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/cleaning/appliances/metadata")
    sys.exit(1)

METADATA_PATH = sys.argv[1]
OUTPUT_BASE   = sys.argv[2]
OUTPUT_DATA   = OUTPUT_BASE + "/cleaned_metadata"
OUTPUT_TXT    = OUTPUT_BASE + "/report"
OUTPUT_CSV    = OUTPUT_BASE + "/summary"

# ─────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("Metadata_Cleaning") \
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
# LOAD WITH EXPLICIT SCHEMA
# ─────────────────────────────────────────
section("STEP 1 — LOADING RAW METADATA")

schema = StructType([
    StructField("parent_asin",     StringType(),  True),
    StructField("title",           StringType(),  True),
    StructField("main_category",   StringType(),  True),
    StructField("store",           StringType(),  True),
    StructField("price",           DoubleType(),  True),
    StructField("average_rating",  DoubleType(),  True),
    StructField("rating_number",   LongType(),    True),
    StructField("features",        ArrayType(StringType()), True),
    StructField("description",     ArrayType(StringType()), True),
    StructField("categories",      ArrayType(StringType()), True),
    StructField("bought_together", StringType(),  True),
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

raw_count = df.count()
log(f"Raw row count          : {raw_count:,}")
log(f"Columns                : {df.columns}")

# ─────────────────────────────────────────
# CLEANING DECISION 1
# Drop rows where parent_asin or title is null
# These cannot be displayed or joined
# ─────────────────────────────────────────
section("STEP 2 — DROP NULL parent_asin OR title")

df_c1 = df.filter(
    F.col("parent_asin").isNotNull() &
    F.col("title").isNotNull() &
    (F.col("title") != "")
)

after_c1 = df_c1.count()
dropped_c1 = raw_count - after_c1
log(f"Rows dropped           : {dropped_c1:,}")
log(f"Rows remaining         : {after_c1:,}")

# ─────────────────────────────────────────
# CLEANING DECISION 2
# Drop rows missing hi_res image URL
# Products without images cannot be shown in the app
# ─────────────────────────────────────────
section("STEP 3 — DROP PRODUCTS WITH NO HI_RES IMAGE")

df_c2 = df_c1.filter(
    F.col("images").isNotNull() &
    (F.size(F.col("images")) > 0) &
    F.col("images")[0]["hi_res"].isNotNull() &
    (F.col("images")[0]["hi_res"] != "")
)

after_c2 = df_c2.count()
dropped_c2 = after_c1 - after_c2
log(f"Rows dropped           : {dropped_c2:,}")
log(f"Rows remaining         : {after_c2:,}")

# ─────────────────────────────────────────
# CLEANING DECISION 3
# Normalize store (brand) name
# Standardize case — trim whitespace
# Fix known duplicates like FRIGIDAIRE vs Frigidaire
# ─────────────────────────────────────────
section("STEP 4 — NORMALIZE STORE NAME")

log("Before normalization — top brand variants:")
df_c2.filter(
    F.lower(F.col("store")).contains("frigidaire") |
    F.lower(F.col("store")).contains("whirlpool") |
    F.lower(F.col("store")).contains("samsung")
).groupBy("store").count() \
 .orderBy(F.col("count").desc()) \
 .show(10, truncate=False)

df_c3 = df_c2.withColumn(
    "store",
    F.when(F.col("store").isNull(), "Unknown")
     .otherwise(F.initcap(F.trim(F.col("store"))))
)

log("After normalization — same brands:")
df_c3.filter(
    F.lower(F.col("store")).contains("frigidaire") |
    F.lower(F.col("store")).contains("whirlpool") |
    F.lower(F.col("store")).contains("samsung")
).groupBy("store").count() \
 .orderBy(F.col("count").desc()) \
 .show(10, truncate=False)

# ─────────────────────────────────────────
# CLEANING DECISION 4
# Extract leaf subcategory from categories array
# Use last element as most specific category
# ─────────────────────────────────────────
section("STEP 5 — EXTRACT LEAF SUBCATEGORY")

df_c4 = df_c3.withColumn(
    "subcategory",
    F.when(
        F.col("categories").isNotNull() & (F.size(F.col("categories")) > 0),
        F.element_at(F.col("categories"), -1)
    ).otherwise("Unknown")
)

log("Leaf subcategory distribution (top 10):")
df_c4.groupBy("subcategory").count() \
     .orderBy(F.col("count").desc()) \
     .show(10, truncate=False)

# ─────────────────────────────────────────
# CLEANING DECISION 5
# Extract hi_res image URL from images array
# Take the MAIN variant image if available
# Otherwise take first image
# ─────────────────────────────────────────
section("STEP 6 — EXTRACT IMAGE URL")

df_c5 = df_c4.withColumn(
    "image_url",
    F.coalesce(
        F.filter(F.col("images"), lambda x: x["variant"] == "MAIN")[0]["hi_res"],
        F.col("images")[0]["hi_res"]
    )
)

has_image = df_c5.filter(F.col("image_url").isNotNull()).count()
log(f"Products with image URL : {has_image:,}")

# ─────────────────────────────────────────
# CLEANING DECISION 6
# Handle missing price
# Keep null as-is but flag it
# Do not impute — unknown price is better than wrong price
# ─────────────────────────────────────────
section("STEP 7 — PRICE HANDLING")

price_null  = df_c5.filter(F.col("price").isNull()).count()
price_valid = df_c5.filter(F.col("price").isNotNull()).count()

log(f"Products with price     : {price_valid:,}  ({round(100*price_valid/after_c2,2)}%)")
log(f"Products without price  : {price_null:,}  ({round(100*price_null/after_c2,2)}%)")
log()
log("NOTE: Null prices are kept as null.")
log("Price will be displayed as 'Price not available' in the app.")

# ─────────────────────────────────────────
# CLEANING DECISION 7
# Deduplicate by parent_asin
# Keep the record with highest rating_number
# More ratings = more reliable product info
# ─────────────────────────────────────────
section("STEP 8 — DEDUPLICATE BY parent_asin")

from pyspark.sql.window import Window

window = Window.partitionBy("parent_asin") \
               .orderBy(F.col("rating_number").desc())

df_c6 = df_c5.withColumn("row_num", F.row_number().over(window)) \
             .filter(F.col("row_num") == 1) \
             .drop("row_num")

after_c6 = df_c6.count()
dropped_c6 = after_c2 - after_c6
log(f"Duplicate rows removed  : {dropped_c6:,}")
log(f"Rows remaining          : {after_c6:,}")

# ─────────────────────────────────────────
# CLEANING DECISION 8
# Strip HTML from title and features
# ─────────────────────────────────────────
section("STEP 9 — STRIP HTML FROM TITLE")

df_c7 = df_c6.withColumn(
    "title",
    F.trim(F.regexp_replace(F.col("title"), "<[^>]+>", " "))
).withColumn(
    "title",
    F.regexp_replace(F.col("title"), "&amp;",  "&")
).withColumn(
    "title",
    F.regexp_replace(F.col("title"), "&lt;",   "<")
).withColumn(
    "title",
    F.regexp_replace(F.col("title"), "&gt;",   ">")
).withColumn(
    "title",
    F.regexp_replace(F.col("title"), "&#[0-9]+;", "")
)

log("HTML stripped from title field.")

# ─────────────────────────────────────────
# CLEANING DECISION 9
# Select final columns needed downstream
# Drop bought_together, raw images array,
# raw categories array, description
# ─────────────────────────────────────────
section("STEP 10 — SELECT FINAL COLUMNS")

df_clean = df_c7.select(
    F.col("parent_asin"),
    F.col("title"),
    F.col("main_category"),
    F.col("subcategory"),
    F.col("store").alias("brand"),
    F.col("price"),
    F.col("average_rating"),
    F.col("rating_number"),
    F.col("image_url"),
    F.col("features"),
)

log("Final columns:")
for col in df_clean.columns:
    log(f"  {col}")

# ─────────────────────────────────────────
# FINAL VALIDATION
# ─────────────────────────────────────────
section("STEP 11 — FINAL VALIDATION")

final_count = df_clean.count()

log(f"Raw row count          : {raw_count:,}")
log(f"Clean row count        : {final_count:,}")
log(f"Total rows removed     : {raw_count - final_count:,}")
log(f"Retention rate         : {round(100*final_count/raw_count, 2)}%")

log()
log("Breakdown of rows removed:")
log(f"  Null parent_asin or title    : {dropped_c1:,}")
log(f"  Missing hi_res image         : {dropped_c2:,}")
log(f"  Duplicate parent_asin        : {dropped_c6:,}")

log()
log("Null check on clean dataset:")
null_check = df_clean.select([
    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df_clean.columns
]).collect()[0]

log(f"\n{'Column':<25} {'Null Count':>12}")
log("-" * 40)
for c in df_clean.columns:
    log(f"{c:<25} {null_check[c]:>12,}")

log()
log("Sample clean records (2 rows):")
sample = df_clean.limit(2).collect()
for i, row in enumerate(sample):
    log(f"\n  Record {i+1}:")
    log(f"    parent_asin    : {row['parent_asin']}")
    log(f"    title          : {str(row['title'])[:80]}")
    log(f"    main_category  : {row['main_category']}")
    log(f"    subcategory    : {row['subcategory']}")
    log(f"    brand          : {row['brand']}")
    log(f"    price          : {row['price']}")
    log(f"    average_rating : {row['average_rating']}")
    log(f"    image_url      : {str(row['image_url'])[:60]}")

# ─────────────────────────────────────────
# SAVE CLEAN DATA
# ─────────────────────────────────────────
section("STEP 12 — SAVING CLEAN DATA")

df_clean.write.mode("overwrite").parquet(OUTPUT_DATA)

log(f"Clean data saved as Parquet.")
log(f"Location : {OUTPUT_DATA}")
log(f"Rows     : {final_count:,}")

summary_data = [
    ("raw_row_count",               str(raw_count)),
    ("clean_row_count",             str(final_count)),
    ("total_rows_removed",          str(raw_count - final_count)),
    ("retention_rate_pct",          str(round(100*final_count/raw_count, 2))),
    ("dropped_null_title",          str(dropped_c1)),
    ("dropped_no_image",            str(dropped_c2)),
    ("dropped_duplicates",          str(dropped_c6)),
    ("price_null_pct",              str(round(100*price_null/after_c2, 2))),
    ("final_columns",               str(df_clean.columns)),
]

summary_df = spark.createDataFrame(summary_data, ["metric", "value"])
summary_df.coalesce(1).write.mode("overwrite") \
    .option("header", "true").csv(OUTPUT_CSV)

report_df = spark.createDataFrame([(l,) for l in lines], ["line"])
report_df.coalesce(1).write.mode("overwrite").text(OUTPUT_TXT)

log()
log("Summary CSV saved.")
log("Full report saved.")
log()
log("=" * 60)
log("CLEANING COMPLETE")
log("=" * 60)

spark.stop()