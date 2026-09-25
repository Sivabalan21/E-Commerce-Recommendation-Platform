import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ─────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────
if len(sys.argv) != 4:
    print("Usage: spark-submit 05_join.py <reviews_path> <metadata_path> <output_path>")
    print("Example:")
    print("  spark-submit 05_join.py \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/cleaning/appliances/reviews/cleaned_reviews \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/cleaning/appliances/metadata/cleaned_metadata \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/joined/appliances")
    sys.exit(1)

REVIEWS_PATH  = sys.argv[1]
METADATA_PATH = sys.argv[2]
OUTPUT_BASE   = sys.argv[3]
OUTPUT_DATA   = OUTPUT_BASE + "/joined_data"
OUTPUT_TXT    = OUTPUT_BASE + "/report"
OUTPUT_CSV    = OUTPUT_BASE + "/summary"

# ─────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("Join_Reviews_Metadata") \
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
# LOAD BOTH CLEAN DATASETS
# ─────────────────────────────────────────
section("STEP 1 — LOADING CLEAN DATASETS")

df_reviews  = spark.read.parquet(REVIEWS_PATH)
df_metadata = spark.read.parquet(METADATA_PATH)

reviews_count  = df_reviews.count()
metadata_count = df_metadata.count()

log(f"Reviews rows   : {reviews_count:,}")
log(f"Metadata rows  : {metadata_count:,}")
log(f"Reviews cols   : {df_reviews.columns}")
log(f"Metadata cols  : {df_metadata.columns}")

# ─────────────────────────────────────────
# JOIN ON parent_asin
# Left join — keep all reviews
# Drop reviews whose product has no metadata
# ─────────────────────────────────────────
section("STEP 2 — JOIN ON parent_asin")

df_joined = df_reviews.join(
    df_metadata,
    on="parent_asin",
    how="left"
)

joined_count = df_joined.count()
log(f"Rows after left join   : {joined_count:,}")

no_metadata = df_joined.filter(F.col("title").isNull()).count()
log(f"Reviews with no metadata match : {no_metadata:,}")
log(f"  ({round(100*no_metadata/joined_count, 2)}% of reviews have no product metadata)")

df_joined = df_joined.filter(F.col("title").isNotNull())
final_count = df_joined.count()
dropped = joined_count - final_count
log(f"Dropped reviews with no metadata : {dropped:,}")
log(f"Final joined row count           : {final_count:,}")

# ─────────────────────────────────────────
# VERIFY JOIN QUALITY
# ─────────────────────────────────────────
section("STEP 3 — JOIN QUALITY CHECK")

log(f"Final columns : {df_joined.columns}")
log()

null_check = df_joined.select([
    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in ["user_id", "parent_asin", "rating",
              "review_text", "title", "brand",
              "subcategory", "image_url"]
]).collect()[0]

log(f"{'Column':<25} {'Null Count':>12}")
log("-" * 40)
for c in ["user_id", "parent_asin", "rating",
          "review_text", "title", "brand",
          "subcategory", "image_url"]:
    log(f"{c:<25} {null_check[c]:>12,}")

log()
log("Sample joined records (2 rows):")
sample = df_joined.limit(2).collect()
for i, row in enumerate(sample):
    log(f"\n  Record {i+1}:")
    log(f"    user_id      : {row['user_id']}")
    log(f"    parent_asin  : {row['parent_asin']}")
    log(f"    rating       : {row['rating']}")
    log(f"    review_text  : {str(row['review_text'])[:80]}")
    log(f"    title        : {str(row['title'])[:80]}")
    log(f"    brand        : {row['brand']}")
    log(f"    subcategory  : {row['subcategory']}")
    log(f"    price        : {row['price']}")
    log(f"    image_url    : {str(row['image_url'])[:60]}")

# ─────────────────────────────────────────
# COVERAGE ANALYSIS
# ─────────────────────────────────────────
section("STEP 4 — COVERAGE ANALYSIS")

unique_users    = df_joined.select(
    F.approx_count_distinct("user_id")).collect()[0][0]
unique_products = df_joined.select(
    F.approx_count_distinct("parent_asin")).collect()[0][0]

log(f"Unique users in joined table    : {unique_users:,}")
log(f"Unique products in joined table : {unique_products:,}")
log()

log("Reviews per main_category (top 10):")
log(f"{'main_category':<40} {'reviews':>10} {'percentage':>12}")
log("-" * 64)
cat_counts = df_joined.groupBy("main_category") \
                      .count() \
                      .orderBy(F.col("count").desc()) \
                      .limit(10).collect()
for row in cat_counts:
    cat = str(row["main_category"])[:39]
    pct = round(100 * row["count"] / final_count, 2)
    log(f"{cat:<40} {row['count']:>10,} {pct:>11}%")

log()
log("Reviews per subcategory (top 10):")
log(f"{'subcategory':<40} {'reviews':>10}")
log("-" * 52)
sub_counts = df_joined.groupBy("subcategory") \
                      .count() \
                      .orderBy(F.col("count").desc()) \
                      .limit(10).collect()
for row in sub_counts:
    sub = str(row["subcategory"])[:39]
    log(f"{sub:<40} {row['count']:>10,}")

log()
log("Reviews per brand (top 10):")
log(f"{'brand':<35} {'reviews':>10}")
log("-" * 46)
brand_counts = df_joined.groupBy("brand") \
                        .count() \
                        .orderBy(F.col("count").desc()) \
                        .limit(10).collect()
for row in brand_counts:
    brand = str(row["brand"])[:34]
    log(f"{brand:<35} {row['count']:>10,}")

# ─────────────────────────────────────────
# REGISTER AS TEMP VIEW FOR SPARK SQL
# ─────────────────────────────────────────
section("STEP 5 — REGISTER AS SPARK SQL VIEW")

df_joined.createOrReplaceTempView("reviews_products")

log("Registered as temp view: reviews_products")
log()
log("Running sample Spark SQL query:")
log("  SELECT main_category, COUNT(*) as reviews, ROUND(AVG(rating),2) as avg_rating")
log("  FROM reviews_products GROUP BY main_category ORDER BY reviews DESC")
log()

sql_result = spark.sql("""
    SELECT
        main_category,
        COUNT(*) as reviews,
        ROUND(AVG(rating), 2) as avg_rating,
        ROUND(AVG(helpful_vote), 2) as avg_helpful_votes
    FROM reviews_products
    GROUP BY main_category
    ORDER BY reviews DESC
""").collect()

log(f"{'main_category':<35} {'reviews':>10} {'avg_rating':>12} {'avg_helpful':>14}")
log("-" * 74)
for row in sql_result:
    cat = str(row["main_category"])[:34]
    log(f"{cat:<35} {row['reviews']:>10,} {row['avg_rating']:>12} {row['avg_helpful_votes']:>14}")

# ─────────────────────────────────────────
# SAVE JOINED DATA
# ─────────────────────────────────────────
section("STEP 6 — SAVING JOINED DATA")

df_joined.write.mode("overwrite").parquet(OUTPUT_DATA)

log(f"Joined data saved as Parquet.")
log(f"Location : {OUTPUT_DATA}")
log(f"Rows     : {final_count:,}")

summary_data = [
    ("reviews_input_count",       str(reviews_count)),
    ("metadata_input_count",      str(metadata_count)),
    ("joined_count",              str(final_count)),
    ("reviews_no_metadata",       str(dropped)),
    ("unique_users",              str(unique_users)),
    ("unique_products",           str(unique_products)),
    ("retention_pct",             str(round(100*final_count/reviews_count, 2))),
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
log("JOIN COMPLETE")
log("=" * 60)

spark.stop()