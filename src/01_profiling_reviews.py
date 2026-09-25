import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ─────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────
if len(sys.argv) != 3:
    print("Usage: spark-submit 01_profiling_reviews.py <input_path> <output_path>")
    print("Example:")
    print("  spark-submit 01_profiling_reviews.py \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/raw/appliances_reviews.json.gz \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/profiling/appliances/reviews")
    sys.exit(1)

REVIEWS_PATH = sys.argv[1]
OUTPUT_BASE  = sys.argv[2]
OUTPUT_CSV   = OUTPUT_BASE + "/summary"
OUTPUT_TXT   = OUTPUT_BASE + "/report"

# ─────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("Reviews_Profiling") \
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
section("STEP 1 — LOADING REVIEWS DATASET")

df = spark.read.json(REVIEWS_PATH)
df.cache()

total_rows = df.count()
log(f"\nTotal rows     : {total_rows:,}")
log(f"Total columns  : {len(df.columns)}")
log(f"Columns        : {df.columns}")

# ─────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────
section("STEP 2 — SCHEMA")

for field in df.schema.fields:
    log(f"  {field.name:<25} {str(field.dataType):<20} nullable={field.nullable}")

# ─────────────────────────────────────────
# NULL ANALYSIS — single pass
# ─────────────────────────────────────────
section("STEP 3 — NULL AND MISSING VALUE ANALYSIS")

cols = ["user_id", "asin", "parent_asin", "rating", "text",
        "verified_purchase", "helpful_vote", "timestamp", "title"]

null_exprs = [
    F.sum(F.when(
        F.col(c).isNull() | (F.col(c).cast("string") == ""), 1
    ).otherwise(0)).alias(c)
    for c in cols if c in df.columns
]
null_counts = df.select(null_exprs).collect()[0]

log(f"\n{'Column':<25} {'Null Count':>12} {'Null %':>10} {'Non-Null':>12}")
log("-" * 62)
for c in cols:
    if c in df.columns:
        nc  = null_counts[c]
        pct = round(100 * nc / total_rows, 4)
        log(f"{c:<25} {nc:>12,} {pct:>9}% {(total_rows-nc):>12,}")

# ─────────────────────────────────────────
# RATING DISTRIBUTION — single pass
# ─────────────────────────────────────────
section("STEP 4 — RATING DISTRIBUTION")

rating_stats = df.select(
    F.count("rating").alias("count"),
    F.min("rating").alias("min"),
    F.max("rating").alias("max"),
    F.round(F.avg("rating"), 4).alias("avg"),
    F.round(F.stddev("rating"), 4).alias("stddev"),
    F.sum(F.when(F.col("rating") < 1.0, 1).otherwise(0)).alias("below_1"),
    F.sum(F.when(F.col("rating") > 5.0, 1).otherwise(0)).alias("above_5"),
    F.sum(F.when(F.col("rating") == 1.0, 1).otherwise(0)).alias("stars_1"),
    F.sum(F.when(F.col("rating") == 2.0, 1).otherwise(0)).alias("stars_2"),
    F.sum(F.when(F.col("rating") == 3.0, 1).otherwise(0)).alias("stars_3"),
    F.sum(F.when(F.col("rating") == 4.0, 1).otherwise(0)).alias("stars_4"),
    F.sum(F.when(F.col("rating") == 5.0, 1).otherwise(0)).alias("stars_5"),
).collect()[0]

log(f"Min rating        : {rating_stats['min']}")
log(f"Max rating        : {rating_stats['max']}")
log(f"Avg rating        : {rating_stats['avg']}")
log(f"Stddev rating     : {rating_stats['stddev']}")
log(f"Invalid (< 1.0)   : {rating_stats['below_1']:,}")
log(f"Invalid (> 5.0)   : {rating_stats['above_5']:,}")
log()
log(f"{'Stars':<10} {'Count':>12} {'Percentage':>12}")
log("-" * 36)
for star in [1, 2, 3, 4, 5]:
    count = rating_stats[f"stars_{star}"]
    pct   = round(100 * count / total_rows, 2)
    log(f"{star} star{'s' if star > 1 else ' ':<5} {count:>12,} {pct:>11}%")

# ─────────────────────────────────────────
# VERIFIED PURCHASE — single pass
# ─────────────────────────────────────────
section("STEP 5 — VERIFIED PURCHASE BREAKDOWN")

verified_stats = df.select(
    F.sum(F.when(F.col("verified_purchase") == True,  1).otherwise(0)).alias("verified"),
    F.sum(F.when(F.col("verified_purchase") == False, 1).otherwise(0)).alias("not_verified"),
    F.sum(F.when(F.col("verified_purchase").isNull(),  1).otherwise(0)).alias("null_verified"),
).collect()[0]

v  = verified_stats["verified"]
nv = verified_stats["not_verified"]
log(f"Verified purchases    : {v:>10,}  ({round(100*v/total_rows,2)}%)")
log(f"Unverified purchases  : {nv:>10,}  ({round(100*nv/total_rows,2)}%)")
log(f"Null verified field   : {verified_stats['null_verified']:>10,}")

# ─────────────────────────────────────────
# UNIQUE COUNTS — single pass
# ─────────────────────────────────────────
section("STEP 6 — UNIQUE COUNTS (approx_count_distinct)")

unique_stats = df.select(
    F.approx_count_distinct("user_id").alias("unique_users"),
    F.approx_count_distinct("asin").alias("unique_asins"),
    F.approx_count_distinct("parent_asin").alias("unique_parent_asins"),
).collect()[0]

log(f"Unique users          : {unique_stats['unique_users']:>10,}")
log(f"Unique ASINs          : {unique_stats['unique_asins']:>10,}")
log(f"Unique parent ASINs   : {unique_stats['unique_parent_asins']:>10,}")

# ─────────────────────────────────────────
# TEXT ANALYSIS — single pass
# ─────────────────────────────────────────
section("STEP 7 — REVIEW TEXT ANALYSIS")

text_stats = df.select(
    F.min(F.length("text")).alias("min_len"),
    F.max(F.length("text")).alias("max_len"),
    F.round(F.avg(F.length("text")), 2).alias("avg_len"),
    F.round(F.stddev(F.length("text")), 2).alias("stddev_len"),
    F.sum(F.when(F.col("text").isNull(), 1)
            .otherwise(0)).alias("null_text"),
    F.sum(F.when(
        F.col("text").isNotNull() &
        (F.regexp_extract(F.col("text"), "[a-zA-Z]", 0) == ""),
        1).otherwise(0)).alias("no_alpha_text"),
    F.sum(F.when(
        F.col("text").contains("<br />") |
        F.col("text").contains("<br/>") |
        F.col("text").contains("<p>"), 1)
        .otherwise(0)).alias("has_html"),
).collect()[0]

log(f"Min text length       : {text_stats['min_len']}")
log(f"Max text length       : {text_stats['max_len']:,}")
log(f"Avg text length       : {text_stats['avg_len']} chars")
log(f"Stddev text length    : {text_stats['stddev_len']} chars")
log(f"Null text rows        : {text_stats['null_text']:,}")
log(f"No alphabetic chars   : {text_stats['no_alpha_text']:,}  (drop criterion)")
log(f"Contains HTML tags    : {text_stats['has_html']:,}  (strip in cleaning)")

log()
log(f"{'Bucket':<20} {'Count':>12} {'Percentage':>12}")
log("-" * 46)

bucket_counts = df.select(
    F.when(F.col("text").isNull(), "null")
     .when(F.length(F.col("text")) == 0,   "0 chars")
     .when(F.length(F.col("text")) < 20,   "1-19 chars")
     .when(F.length(F.col("text")) < 100,  "20-99 chars")
     .when(F.length(F.col("text")) < 500,  "100-499 chars")
     .when(F.length(F.col("text")) < 1000, "500-999 chars")
     .otherwise("1000+ chars")
     .alias("bucket")
).groupBy("bucket").count().orderBy("bucket").collect()

for row in bucket_counts:
    pct = round(100 * row["count"] / total_rows, 2)
    log(f"{row['bucket']:<20} {row['count']:>12,} {pct:>11}%")

# ─────────────────────────────────────────
# TEMPORAL COVERAGE
# ─────────────────────────────────────────
section("STEP 8 — TEMPORAL COVERAGE")

df_time = df.withColumn(
    "review_date", F.to_timestamp(F.col("timestamp") / 1000)
)

time_stats = df_time.select(
    F.min("review_date").alias("oldest"),
    F.max("review_date").alias("newest"),
).collect()[0]

log(f"Oldest review : {time_stats['oldest']}")
log(f"Newest review : {time_stats['newest']}")
log()
log(f"{'Year':<8} {'Count':>12} {'Percentage':>12}")
log("-" * 34)

year_counts = df_time.withColumn("year", F.year("review_date")) \
                     .groupBy("year").count() \
                     .orderBy("year").collect()

for row in year_counts:
    pct = round(100 * row["count"] / total_rows, 2)
    log(f"{row['year']:<8} {row['count']:>12,} {pct:>11}%")

# ─────────────────────────────────────────
# DUPLICATE DETECTION
# ─────────────────────────────────────────
section("STEP 9 — DUPLICATE DETECTION")

exact_dupes      = total_rows - df.dropDuplicates(["user_id", "asin", "timestamp"]).count()
user_asin_dupes  = df.groupBy("user_id", "asin").count() \
                     .filter(F.col("count") > 1).count()

log(f"Exact duplicates (user+asin+timestamp) : {exact_dupes:,}")
log(f"Same user reviewed same product twice  : {user_asin_dupes:,}")

log()
log(f"{'user_id':<35} {'asin':<15} {'count':>6}")
log("-" * 58)
top_dupes = df.groupBy("user_id", "asin").count() \
              .filter(F.col("count") > 1) \
              .orderBy(F.col("count").desc()) \
              .limit(5).collect()
for row in top_dupes:
    log(f"{row['user_id']:<35} {row['asin']:<15} {row['count']:>6}")

# ─────────────────────────────────────────
# HELPFUL VOTE ANALYSIS — single pass
# ─────────────────────────────────────────
section("STEP 10 — HELPFUL VOTE ANALYSIS")

vote_stats = df.select(
    F.min("helpful_vote").alias("min"),
    F.max("helpful_vote").alias("max"),
    F.round(F.avg("helpful_vote"), 4).alias("avg"),
    F.sum("helpful_vote").alias("total"),
    F.sum(F.when(F.col("helpful_vote") == 0,   1).otherwise(0)).alias("zero_votes"),
    F.sum(F.when(F.col("helpful_vote") >  10,  1).otherwise(0)).alias("over_10"),
    F.sum(F.when(F.col("helpful_vote") >  100, 1).otherwise(0)).alias("over_100"),
).collect()[0]

log(f"Min helpful votes     : {vote_stats['min']}")
log(f"Max helpful votes     : {vote_stats['max']:,}")
log(f"Avg helpful votes     : {vote_stats['avg']}")
log(f"Total helpful votes   : {vote_stats['total']:,}")
log(f"Zero vote reviews     : {vote_stats['zero_votes']:,}  ({round(100*vote_stats['zero_votes']/total_rows,2)}%)")
log(f"More than 10 votes    : {vote_stats['over_10']:,}")
log(f"More than 100 votes   : {vote_stats['over_100']:,}")

# ─────────────────────────────────────────
# ASIN vs PARENT_ASIN
# ─────────────────────────────────────────
section("STEP 11 — ASIN vs PARENT_ASIN ANALYSIS")

asin_stats = df.select(
    F.sum(F.when(F.col("asin") == F.col("parent_asin"), 1)
            .otherwise(0)).alias("same"),
    F.sum(F.when(F.col("asin") != F.col("parent_asin"), 1)
            .otherwise(0)).alias("different"),
).collect()[0]

same = asin_stats["same"]
diff = asin_stats["different"]
log(f"ASIN equals parent_asin   : {same:>10,}  ({round(100*same/total_rows,2)}%)")
log(f"ASIN differs parent_asin  : {diff:>10,}  ({round(100*diff/total_rows,2)}%)")
log()
log("NOTE: ALS will use parent_asin to group product variants.")

# ─────────────────────────────────────────
# TOP PRODUCTS AND USERS
# ─────────────────────────────────────────
section("STEP 12 — TOP REVIEWED PRODUCTS AND MOST ACTIVE USERS")

log("Top 10 most reviewed products (parent_asin):")
log(f"{'parent_asin':<15} {'review_count':>14}")
log("-" * 30)
top_products = df.groupBy("parent_asin").count() \
                 .orderBy(F.col("count").desc()) \
                 .limit(10).collect()
for row in top_products:
    log(f"{row['parent_asin']:<15} {row['count']:>14,}")

log()
log("Top 10 most active reviewers:")
log(f"{'user_id':<35} {'review_count':>14}")
log("-" * 50)
top_users = df.groupBy("user_id").count() \
              .orderBy(F.col("count").desc()) \
              .limit(10).collect()
for row in top_users:
    log(f"{row['user_id']:<35} {row['count']:>14,}")

log()
log("Reviews per user statistics:")
reviews_per_user = df.groupBy("user_id").count().select(
    F.min("count").alias("min"),
    F.max("count").alias("max"),
    F.round(F.avg("count"), 2).alias("avg"),
    F.percentile_approx("count", 0.5).alias("median"),
    F.percentile_approx("count", 0.95).alias("p95"),
).collect()[0]

log(f"Min reviews per user  : {reviews_per_user['min']}")
log(f"Max reviews per user  : {reviews_per_user['max']:,}")
log(f"Avg reviews per user  : {reviews_per_user['avg']}")
log(f"Median reviews/user   : {reviews_per_user['median']}")
log(f"95th pct reviews/user : {reviews_per_user['p95']}")

# ─────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────
section("STEP 13 — SAVING OUTPUTS")

summary_data = [
    ("total_rows",                    str(total_rows)),
    ("total_columns",                 str(len(df.columns))),
    ("unique_users",                  str(unique_stats["unique_users"])),
    ("unique_asins",                  str(unique_stats["unique_asins"])),
    ("unique_parent_asins",           str(unique_stats["unique_parent_asins"])),
    ("avg_rating",                    str(rating_stats["avg"])),
    ("pct_5_star",                    str(round(100*rating_stats["stars_5"]/total_rows,2))),
    ("pct_1_star",                    str(round(100*rating_stats["stars_1"]/total_rows,2))),
    ("invalid_ratings",               str(rating_stats["below_1"] + rating_stats["above_5"])),
    ("pct_verified_purchase",         str(round(100*v/total_rows,2))),
    ("null_text_rows",                str(text_stats["null_text"])),
    ("no_alpha_text_rows",            str(text_stats["no_alpha_text"])),
    ("has_html_in_text",              str(text_stats["has_html"])),
    ("avg_text_length_chars",         str(text_stats["avg_len"])),
    ("exact_duplicates",              str(exact_dupes)),
    ("user_asin_duplicate_pairs",     str(user_asin_dupes)),
    ("zero_helpful_vote_reviews",     str(vote_stats["zero_votes"])),
    ("asin_equals_parent_asin",       str(same)),
    ("asin_differs_parent_asin",      str(diff)),
    ("oldest_review",                 str(time_stats["oldest"])),
    ("newest_review",                 str(time_stats["newest"])),
]

summary_df = spark.createDataFrame(summary_data, ["metric", "value"])
summary_df.coalesce(1).write.mode("overwrite") \
    .option("header", "true").csv(OUTPUT_CSV)

report_df = spark.createDataFrame([(l,) for l in lines], ["line"])
report_df.coalesce(1).write.mode("overwrite").text(OUTPUT_TXT)

log(f"Summary CSV : {OUTPUT_CSV}")
log(f"Full report : {OUTPUT_TXT}")
log()
log("=" * 60)
log("PROFILING COMPLETE")
log("=" * 60)

spark.stop()