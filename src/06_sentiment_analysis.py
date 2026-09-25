import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, FloatType, StructType, StructField

# ─────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────
if len(sys.argv) != 3:
    print("Usage: spark-submit 06_sentiment_analysis.py <input_path> <output_path>")
    print("Example:")
    print("  spark-submit 06_sentiment_analysis.py \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/joined/appliances/joined_data \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/sentiment/appliances")
    sys.exit(1)

INPUT_PATH  = sys.argv[1]
OUTPUT_BASE = sys.argv[2]
OUTPUT_DATA = OUTPUT_BASE + "/sentiment_data"
OUTPUT_TXT  = OUTPUT_BASE + "/report"
OUTPUT_CSV  = OUTPUT_BASE + "/summary"

# ─────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("Sentiment_Analysis") \
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
# LOAD JOINED DATA
# ─────────────────────────────────────────
section("STEP 1 — LOADING JOINED DATA")

df = spark.read.parquet(INPUT_PATH)
df.cache()

total_rows = df.count()
log(f"Total rows     : {total_rows:,}")
log(f"Columns        : {df.columns}")

# ─────────────────────────────────────────
# APPLY SENTIMENT ANALYSIS
# mapPartitions creates one analyzer per partition
# not one per row — much more efficient
# ─────────────────────────────────────────
section("STEP 2 — APPLYING VADER SENTIMENT ANALYSIS")

log("Applying VADER to review_text column...")
log("Using mapPartitions — one analyzer instance per partition...")
log("Scoring thresholds:")
log("  Positive : compound score >= +0.05")
log("  Negative : compound score <= -0.05")
log("  Neutral  : compound score between -0.05 and +0.05")

def analyze_partition(rows):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    for row in rows:
        text = row["review_text"]
        if text is None or str(text).strip() == "":
            score = 0.0
            label = "Neutral"
        else:
            score = float(analyzer.polarity_scores(str(text))["compound"])
            if score >= 0.05:
                label = "Positive"
            elif score <= -0.05:
                label = "Negative"
            else:
                label = "Neutral"
        yield (*row, score, label)

output_schema = StructType(
    df.schema.fields + [
        StructField("sentiment_score", FloatType(),  True),
        StructField("sentiment_label", StringType(), True),
    ]
)

df_sentiment = spark.createDataFrame(
    df.rdd.mapPartitions(analyze_partition),
    schema=output_schema
)

df_sentiment.cache()
scored_count = df_sentiment.count()
log(f"Sentiment scored rows : {scored_count:,}")

# ─────────────────────────────────────────
# SENTIMENT DISTRIBUTION
# ─────────────────────────────────────────
section("STEP 3 — SENTIMENT DISTRIBUTION")

sentiment_dist = df_sentiment.groupBy("sentiment_label") \
    .count() \
    .orderBy("sentiment_label") \
    .collect()

log(f"{'Label':<12} {'Count':>12} {'Percentage':>12}")
log("-" * 38)
for row in sentiment_dist:
    pct = round(100 * row["count"] / scored_count, 2)
    log(f"{row['sentiment_label']:<12} {row['count']:>12,} {pct:>11}%")

# ─────────────────────────────────────────
# SENTIMENT vs RATING ANALYSIS
# ─────────────────────────────────────────
section("STEP 4 — SENTIMENT vs RATING ANALYSIS")

log("Average sentiment score per star rating:")
log(f"{'Rating':<10} {'Avg Sentiment':>15} {'Count':>10}")
log("-" * 38)

rating_sentiment = df_sentiment.groupBy("rating").agg(
    F.round(F.avg("sentiment_score"), 4).alias("avg_sentiment"),
    F.count("*").alias("count")
).orderBy("rating").collect()

for row in rating_sentiment:
    log(f"{row['rating']:<10} {row['avg_sentiment']:>15} {row['count']:>10,}")

log()
log("Disagreement analysis:")
disagree_pos = df_sentiment.filter(
    (F.col("rating") >= 4.0) &
    (F.col("sentiment_label") == "Negative")
).count()

disagree_neg = df_sentiment.filter(
    (F.col("rating") <= 2.0) &
    (F.col("sentiment_label") == "Positive")
).count()

log(f"High rating (4-5 stars) but Negative sentiment : {disagree_pos:,}  ({round(100*disagree_pos/scored_count,2)}%)")
log(f"Low rating (1-2 stars) but Positive sentiment  : {disagree_neg:,}  ({round(100*disagree_neg/scored_count,2)}%)")

# ─────────────────────────────────────────
# SENTIMENT BY CATEGORY
# ─────────────────────────────────────────
section("STEP 5 — SENTIMENT BY MAIN CATEGORY")

log(f"{'main_category':<35} {'avg_score':>10} {'positive%':>10} {'negative%':>10} {'reviews':>10}")
log("-" * 78)

cat_sentiment = df_sentiment.groupBy("main_category").agg(
    F.round(F.avg("sentiment_score"), 4).alias("avg_score"),
    F.round(F.sum(F.when(F.col("sentiment_label") == "Positive", 1).otherwise(0)) * 100.0 / F.count("*"), 2).alias("positive_pct"),
    F.round(F.sum(F.when(F.col("sentiment_label") == "Negative", 1).otherwise(0)) * 100.0 / F.count("*"), 2).alias("negative_pct"),
    F.count("*").alias("reviews")
).orderBy(F.col("reviews").desc()).limit(10).collect()

for row in cat_sentiment:
    cat = str(row["main_category"])[:34]
    log(f"{cat:<35} {row['avg_score']:>10} {row['positive_pct']:>9}% {row['negative_pct']:>9}% {row['reviews']:>10,}")

# ─────────────────────────────────────────
# SENTIMENT BY BRAND
# ─────────────────────────────────────────
section("STEP 6 — SENTIMENT BY BRAND (top 15)")

log(f"{'brand':<30} {'avg_score':>10} {'positive%':>10} {'negative%':>10} {'reviews':>10}")
log("-" * 73)

brand_sentiment = df_sentiment.groupBy("brand").agg(
    F.round(F.avg("sentiment_score"), 4).alias("avg_score"),
    F.round(F.sum(F.when(F.col("sentiment_label") == "Positive", 1).otherwise(0)) * 100.0 / F.count("*"), 2).alias("positive_pct"),
    F.round(F.sum(F.when(F.col("sentiment_label") == "Negative", 1).otherwise(0)) * 100.0 / F.count("*"), 2).alias("negative_pct"),
    F.count("*").alias("reviews")
).orderBy(F.col("reviews").desc()).limit(15).collect()

for row in brand_sentiment:
    brand = str(row["brand"])[:29]
    log(f"{brand:<30} {row['avg_score']:>10} {row['positive_pct']:>9}% {row['negative_pct']:>9}% {row['reviews']:>10,}")

# ─────────────────────────────────────────
# SENTIMENT BY SUBCATEGORY
# ─────────────────────────────────────────
section("STEP 7 — SENTIMENT BY SUBCATEGORY (top 15)")

log(f"{'subcategory':<35} {'avg_score':>10} {'positive%':>10} {'negative%':>10} {'reviews':>10}")
log("-" * 78)

sub_sentiment = df_sentiment.groupBy("subcategory").agg(
    F.round(F.avg("sentiment_score"), 4).alias("avg_score"),
    F.round(F.sum(F.when(F.col("sentiment_label") == "Positive", 1).otherwise(0)) * 100.0 / F.count("*"), 2).alias("positive_pct"),
    F.round(F.sum(F.when(F.col("sentiment_label") == "Negative", 1).otherwise(0)) * 100.0 / F.count("*"), 2).alias("negative_pct"),
    F.count("*").alias("reviews")
).orderBy(F.col("reviews").desc()).limit(15).collect()

for row in sub_sentiment:
    sub = str(row["subcategory"])[:34]
    log(f"{sub:<35} {row['avg_score']:>10} {row['positive_pct']:>9}% {row['negative_pct']:>9}% {row['reviews']:>10,}")

# ─────────────────────────────────────────
# TOP AND BOTTOM PRODUCTS BY SENTIMENT
# ─────────────────────────────────────────
section("STEP 8 — TOP AND BOTTOM PRODUCTS BY SENTIMENT")

log("Top 10 products with highest avg sentiment (min 100 reviews):")
log(f"{'parent_asin':<15} {'title':<45} {'avg_score':>10} {'reviews':>8}")
log("-" * 82)

top_products = df_sentiment.groupBy("parent_asin", "title").agg(
    F.round(F.avg("sentiment_score"), 4).alias("avg_score"),
    F.count("*").alias("reviews")
).filter(F.col("reviews") >= 100) \
 .orderBy(F.col("avg_score").desc()) \
 .limit(10).collect()

for row in top_products:
    title = str(row["title"])[:44]
    log(f"{row['parent_asin']:<15} {title:<45} {row['avg_score']:>10} {row['reviews']:>8,}")

log()
log("Bottom 10 products with lowest avg sentiment (min 100 reviews):")
log(f"{'parent_asin':<15} {'title':<45} {'avg_score':>10} {'reviews':>8}")
log("-" * 82)

bottom_products = df_sentiment.groupBy("parent_asin", "title").agg(
    F.round(F.avg("sentiment_score"), 4).alias("avg_score"),
    F.count("*").alias("reviews")
).filter(F.col("reviews") >= 100) \
 .orderBy(F.col("avg_score").asc()) \
 .limit(10).collect()

for row in bottom_products:
    title = str(row["title"])[:44]
    log(f"{row['parent_asin']:<15} {title:<45} {row['avg_score']:>10} {row['reviews']:>8,}")

# ─────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────
section("STEP 9 — SAVING OUTPUTS")

df_sentiment.write.mode("overwrite").parquet(OUTPUT_DATA)
log(f"Sentiment data saved as Parquet.")
log(f"Location : {OUTPUT_DATA}")
log(f"Rows     : {scored_count:,}")

sentiment_counts = {row["sentiment_label"]: row["count"] for row in sentiment_dist}

summary_data = [
    ("total_rows_scored",         str(scored_count)),
    ("positive_count",            str(sentiment_counts.get("Positive", 0))),
    ("neutral_count",             str(sentiment_counts.get("Neutral", 0))),
    ("negative_count",            str(sentiment_counts.get("Negative", 0))),
    ("positive_pct",              str(round(100*sentiment_counts.get("Positive",0)/scored_count,2))),
    ("neutral_pct",               str(round(100*sentiment_counts.get("Neutral",0)/scored_count,2))),
    ("negative_pct",              str(round(100*sentiment_counts.get("Negative",0)/scored_count,2))),
    ("high_rating_neg_sentiment", str(disagree_pos)),
    ("low_rating_pos_sentiment",  str(disagree_neg)),
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
log("SENTIMENT ANALYSIS COMPLETE")
log("=" * 60)

spark.stop()