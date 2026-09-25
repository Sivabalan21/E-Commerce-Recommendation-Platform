import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ─────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────
if len(sys.argv) != 3:
    print("Usage: spark-submit 03_cleaning_reviews.py <input_path> <output_path>")
    print("Example:")
    print("  spark-submit 03_cleaning_reviews.py \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/raw/appliances_reviews.json.gz \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/cleaning/appliances/reviews")
    sys.exit(1)

REVIEWS_PATH = sys.argv[1]
OUTPUT_BASE  = sys.argv[2]
OUTPUT_DATA  = OUTPUT_BASE + "/cleaned_reviews"
OUTPUT_TXT   = OUTPUT_BASE + "/report"
OUTPUT_CSV   = OUTPUT_BASE + "/summary"

# ─────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("Reviews_Cleaning") \
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
section("STEP 1 — LOADING RAW REVIEWS")

df = spark.read.json(REVIEWS_PATH)
df.cache()

raw_count = df.count()
log(f"Raw row count          : {raw_count:,}")
log(f"Columns                : {df.columns}")

# ─────────────────────────────────────────
# CLEANING DECISION 1
# Drop rows where user_id or asin is null
# ─────────────────────────────────────────
section("STEP 2 — DROP NULL user_id OR asin")

df_c1 = df.filter(
    F.col("user_id").isNotNull() &
    F.col("asin").isNotNull() &
    F.col("parent_asin").isNotNull()
)

after_c1 = df_c1.count()
dropped_c1 = raw_count - after_c1
log(f"Rows dropped           : {dropped_c1:,}")
log(f"Rows remaining         : {after_c1:,}")

# ─────────────────────────────────────────
# CLEANING DECISION 2
# Drop rows where rating is null or outside [1.0, 5.0]
# ─────────────────────────────────────────
section("STEP 3 — DROP INVALID RATINGS")

df_c2 = df_c1.filter(
    F.col("rating").isNotNull() &
    (F.col("rating") >= 1.0) &
    (F.col("rating") <= 5.0)
)

after_c2 = df_c2.count()
dropped_c2 = after_c1 - after_c2
log(f"Rows dropped           : {dropped_c2:,}")
log(f"Rows remaining         : {after_c2:,}")

# ─────────────────────────────────────────
# CLEANING DECISION 3
# Drop rows where text has no alphabetic characters
# This removes rows like ".", "123", "!!!"
# but keeps valid short reviews like "Good", "Bad", "Ok"
# ─────────────────────────────────────────
section("STEP 4 — DROP REVIEWS WITH NO ALPHABETIC CHARACTERS")

df_c3 = df_c2.filter(
    F.col("text").isNotNull() &
    (F.regexp_extract(F.col("text"), "[a-zA-Z]", 0) != "")
)

after_c3 = df_c3.count()
dropped_c3 = after_c2 - after_c3
log(f"Rows dropped           : {dropped_c3:,}")
log(f"Rows remaining         : {after_c3:,}")
log()
log("NOTE: Short reviews like 'Good', 'Bad', 'Ok' are kept.")
log("Only rows with zero alphabetic characters are removed.")

# ─────────────────────────────────────────
# CLEANING DECISION 4
# Strip HTML tags from review text
# Some reviews contain <br />, <p>, &amp; etc.
# ─────────────────────────────────────────
section("STEP 5 — STRIP HTML TAGS FROM REVIEW TEXT")

html_count = df_c3.filter(
    F.col("text").contains("<br />") |
    F.col("text").contains("<br/>") |
    F.col("text").contains("<p>") |
    F.col("text").contains("&amp;") |
    F.col("text").contains("&lt;") |
    F.col("text").contains("&gt;")
).count()

log(f"Reviews with HTML tags : {html_count:,}")

df_c4 = df_c3.withColumn(
    "text",
    F.regexp_replace(F.col("text"), "<[^>]+>", " ")
).withColumn(
    "text",
    F.regexp_replace(F.col("text"), "&amp;",  "&")
).withColumn(
    "text",
    F.regexp_replace(F.col("text"), "&lt;",   "<")
).withColumn(
    "text",
    F.regexp_replace(F.col("text"), "&gt;",   ">")
).withColumn(
    "text",
    F.regexp_replace(F.col("text"), "&nbsp;", " ")
).withColumn(
    "text",
    F.trim(F.regexp_replace(F.col("text"), "\\s+", " "))
)

log(f"HTML tags stripped from text field.")

# ─────────────────────────────────────────
# CLEANING DECISION 5
# Deduplicate by (user_id, parent_asin)
# Keep the most recent review per user per product
# Use parent_asin not asin to treat variants as same product
# ─────────────────────────────────────────
section("STEP 6 — DEDUPLICATE BY (user_id, parent_asin)")

window = Window.partitionBy("user_id", "parent_asin") \
               .orderBy(F.col("timestamp").desc())

df_c5 = df_c4.withColumn("row_num", F.row_number().over(window)) \
             .filter(F.col("row_num") == 1) \
             .drop("row_num")

after_c5 = df_c5.count()
dropped_c5 = after_c3 - after_c5
log(f"Duplicate pairs removed: {dropped_c5:,}")
log(f"Rows remaining         : {after_c5:,}")
log()
log("NOTE: Used parent_asin (not asin) so product variants")
log("are treated as the same product for deduplication.")

# ─────────────────────────────────────────
# CLEANING DECISION 6
# Normalize timestamp from milliseconds to seconds
# Add human readable date column
# ─────────────────────────────────────────
section("STEP 7 — NORMALIZE TIMESTAMP")

df_c6 = df_c5.withColumn(
    "timestamp_seconds",
    (F.col("timestamp") / 1000).cast("long")
).withColumn(
    "review_date",
    F.to_date(F.to_timestamp(F.col("timestamp") / 1000))
)

log(f"Added timestamp_seconds : timestamp divided by 1000")
log(f"Added review_date       : human readable date column")

# ─────────────────────────────────────────
# CLEANING DECISION 7
# Select and rename only the columns we need
# Drop images, title (review title not product title)
# ─────────────────────────────────────────
section("STEP 8 — SELECT FINAL COLUMNS")

df_clean = df_c6.select(
    F.col("user_id"),
    F.col("parent_asin"),
    F.col("asin"),
    F.col("rating"),
    F.col("text").alias("review_text"),
    F.col("helpful_vote"),
    F.col("verified_purchase"),
    F.col("timestamp_seconds").alias("timestamp"),
    F.col("review_date"),
)

log(f"Final columns selected:")
for col in df_clean.columns:
    log(f"  {col}")

# ─────────────────────────────────────────
# FINAL VALIDATION
# ─────────────────────────────────────────
section("STEP 9 — FINAL VALIDATION")

final_count = df_clean.count()

log(f"Raw row count          : {raw_count:,}")
log(f"Clean row count        : {final_count:,}")
log(f"Total rows removed     : {raw_count - final_count:,}")
log(f"Retention rate         : {round(100*final_count/raw_count, 2)}%")

log()
log("Breakdown of rows removed:")
log(f"  Null user_id/asin/parent_asin : {dropped_c1:,}")
log(f"  Invalid ratings               : {dropped_c2:,}")
log(f"  No alphabetic text            : {dropped_c3:,}")
log(f"  Duplicate user-product pairs  : {dropped_c5:,}")

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
log("Rating distribution after cleaning:")
log(f"{'Stars':<10} {'Count':>12} {'Percentage':>12}")
log("-" * 36)
rating_dist = df_clean.groupBy("rating") \
                      .count() \
                      .orderBy("rating") \
                      .collect()
for row in rating_dist:
    pct = round(100 * row["count"] / final_count, 2)
    log(f"{row['rating']:<10} {row['count']:>12,} {pct:>11}%")

log()
log("Sample clean records (3 rows):")
sample = df_clean.limit(3).collect()
for i, row in enumerate(sample):
    log(f"\n  Record {i+1}:")
    log(f"    user_id          : {row['user_id']}")
    log(f"    parent_asin      : {row['parent_asin']}")
    log(f"    rating           : {row['rating']}")
    log(f"    review_text      : {str(row['review_text'])[:100]}")
    log(f"    verified_purchase: {row['verified_purchase']}")
    log(f"    review_date      : {row['review_date']}")

# ─────────────────────────────────────────
# SAVE CLEAN DATA
# ─────────────────────────────────────────
section("STEP 10 — SAVING CLEAN DATA")

df_clean.write.mode("overwrite") \
    .parquet(OUTPUT_DATA)

log(f"Clean data saved as Parquet to:")
log(f"  {OUTPUT_DATA}")
log(f"  Format: Parquet (columnar, optimized for Spark)")
log(f"  Rows: {final_count:,}")

summary_data = [
    ("raw_row_count",               str(raw_count)),
    ("clean_row_count",             str(final_count)),
    ("total_rows_removed",          str(raw_count - final_count)),
    ("retention_rate_pct",          str(round(100*final_count/raw_count, 2))),
    ("dropped_null_ids",            str(dropped_c1)),
    ("dropped_invalid_ratings",     str(dropped_c2)),
    ("dropped_no_alpha_text",       str(dropped_c3)),
    ("dropped_duplicates",          str(dropped_c5)),
    ("html_reviews_cleaned",        str(html_count)),
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