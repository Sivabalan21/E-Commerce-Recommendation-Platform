import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ─────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────
if len(sys.argv) != 5:
    print("Usage: spark-submit 10_recommendation_scoring.py "
          "<recommendations_path> <pagerank_path> "
          "<metadata_path> <output_path>")
    sys.exit(1)

RECS_PATH     = sys.argv[1]
PAGERANK_PATH = sys.argv[2]
METADATA_PATH = sys.argv[3]
OUTPUT_BASE   = sys.argv[4]

# ─────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("Recommendation_Scoring") \
    .config("spark.sql.adaptive.enabled",   "true") \
    .config("spark.sql.shuffle.partitions", "200") \
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

def save_csv(df, name):
    path = OUTPUT_BASE + "/" + name
    df.coalesce(1).write.mode("overwrite") \
      .option("header", "true").csv(path)
    log(f"Saved: {name}")

# ─────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────
section("STEP 1 — LOADING DATA")

df_recs = spark.read.parquet(RECS_PATH)
df_pr   = spark.read.parquet(PAGERANK_PATH)
df_meta = spark.read.parquet(METADATA_PATH)

log(f"User recommendations : {df_recs.count():,} users")
log(f"PageRank products    : {df_pr.count():,} products")
log(f"Metadata products    : {df_meta.count():,} products")

# ─────────────────────────────────────────
# STEP 2 — EXPLODE AND MAP TO PARENT_ASIN
# ─────────────────────────────────────────
section("STEP 2 — EXPLODING RECOMMENDATIONS")

from pyspark.ml.feature import StringIndexerModel

product_indexer = StringIndexerModel.load(
    "hdfs:///user/ss21176_nyu_edu/ecommerce/models/appliances/product_indexer"
)

labels   = product_indexer.labels
label_df = spark.createDataFrame(
    [(i, labels[i]) for i in range(len(labels))],
    ["product_index", "parent_asin"]
)

df_exploded = df_recs.select(
    "user_index",
    F.explode("recommendations").alias("rec")
).select(
    "user_index",
    F.col("rec.product_index").alias("product_index"),
    F.col("rec.rating").alias("als_score")
).join(label_df, on="product_index", how="inner")

log(f"Total user-product pairs : {df_exploded.count():,}")

# ─────────────────────────────────────────
# STEP 3 — NORMALIZE AND JOIN
# Register all tables as SQL views
# ─────────────────────────────────────────
section("STEP 3 — REGISTERING SPARK SQL VIEWS")

als_stats = df_exploded.agg(
    F.min("als_score").alias("min_als"),
    F.max("als_score").alias("max_als")
).collect()[0]

min_als = float(als_stats["min_als"])
max_als = float(als_stats["max_als"])

log(f"ALS score range : {min_als:.4f} to {max_als:.4f}")

df_exploded.withColumn(
    "als_normalized",
    F.round((F.col("als_score") - min_als) / (max_als - min_als), 4)
).createOrReplaceTempView("als_recs")

df_pr.select(
    "parent_asin",
    F.col("pagerank_normalized").alias("pagerank_score")
).createOrReplaceTempView("pagerank")

df_meta.select(
    "parent_asin", "title", "brand",
    "subcategory", "main_category",
    "price", "image_url", "average_rating"
).createOrReplaceTempView("metadata")

log("Registered views: als_recs, pagerank, metadata")

# ─────────────────────────────────────────
# STEP 4 — COMPUTE FINAL SCORES
# ─────────────────────────────────────────
section("STEP 4 — COMPUTING FINAL SCORES (Spark SQL)")

df_scored = spark.sql("""
    SELECT
        a.user_index,
        a.parent_asin,
        a.als_normalized,
        COALESCE(p.pagerank_score, 0.0)             AS pagerank_score,
        ROUND(
            0.7 * a.als_normalized +
            0.3 * COALESCE(p.pagerank_score, 0.0),
            4
        )                                           AS final_score
    FROM als_recs a
    LEFT JOIN pagerank p
        ON a.parent_asin = p.parent_asin
""")

df_scored.createOrReplaceTempView("scored_recs")
log("Final score = 0.7 x ALS + 0.3 x PageRank")

score_stats = spark.sql("""
    SELECT
        ROUND(MIN(final_score), 4)    AS min_score,
        ROUND(MAX(final_score), 4)    AS max_score,
        ROUND(AVG(final_score), 4)    AS avg_score,
        ROUND(AVG(als_normalized), 4) AS avg_als,
        ROUND(AVG(pagerank_score), 4) AS avg_pagerank
    FROM scored_recs
""").collect()[0]

log(f"Min final score  : {score_stats['min_score']}")
log(f"Max final score  : {score_stats['max_score']}")
log(f"Avg final score  : {score_stats['avg_score']}")
log(f"Avg ALS score    : {score_stats['avg_als']}")
log(f"Avg PageRank     : {score_stats['avg_pagerank']}")

# ─────────────────────────────────────────
# STEP 5 — ENRICH WITH METADATA
# ─────────────────────────────────────────
section("STEP 5 — ENRICHING WITH METADATA (Spark SQL)")

df_enriched = spark.sql("""
    SELECT
        s.user_index,
        s.parent_asin,
        s.als_normalized,
        s.pagerank_score,
        s.final_score,
        m.title,
        m.brand,
        m.subcategory,
        m.main_category,
        m.price,
        m.image_url,
        m.average_rating
    FROM scored_recs s
    LEFT JOIN metadata m
        ON s.parent_asin = m.parent_asin
""")

df_enriched.cache()
df_enriched.createOrReplaceTempView("enriched_recs")

enriched_count = df_enriched.count()
log(f"Enriched recommendations : {enriched_count:,} rows")

# ─────────────────────────────────────────
# STEP 6 — SAVE FINAL RECOMMENDATIONS
# ─────────────────────────────────────────
section("STEP 6 — SAVING FINAL RECOMMENDATIONS")

df_enriched.write.mode("overwrite").parquet(
    OUTPUT_BASE + "/final_recommendations"
)
log("Final recommendations saved as Parquet.")

# ─────────────────────────────────────────
# ANALYTICS 1 — TOP RECOMMENDED PRODUCTS
# ─────────────────────────────────────────
section("ANALYTICS 1 — TOP RECOMMENDED PRODUCTS")

top_products = spark.sql("""
    SELECT
        parent_asin,
        MAX(title)                          AS title,
        MAX(brand)                          AS brand,
        MAX(subcategory)                    AS subcategory,
        MAX(price)                          AS price,
        MAX(image_url)                      AS image_url,
        MAX(average_rating)                 AS average_rating,
        COUNT(*)                            AS times_recommended,
        ROUND(AVG(final_score), 4)          AS avg_final_score,
        ROUND(AVG(als_normalized), 4)       AS avg_als_score,
        ROUND(AVG(pagerank_score), 4)       AS avg_pagerank_score
    FROM enriched_recs
    GROUP BY parent_asin
    ORDER BY times_recommended DESC
    LIMIT 100
""")

log("Top 10 most recommended products:")
log(f"{'parent_asin':<15} {'title':<40} {'times':>8} {'score':>8}")
log("-" * 74)
for row in top_products.limit(10).collect():
    title = str(row["title"])[:39]
    log(f"{row['parent_asin']:<15} {title:<40} "
        f"{row['times_recommended']:>8,} {row['avg_final_score']:>8}")

save_csv(top_products, "top_products_analytics")

# ─────────────────────────────────────────
# ANALYTICS 2 — BRAND DISTRIBUTION
# ─────────────────────────────────────────
section("ANALYTICS 2 — BRAND DISTRIBUTION")

brand_dist = spark.sql("""
    SELECT
        brand,
        COUNT(*)                            AS recommendation_count,
        ROUND(COUNT(*) * 100.0
              / SUM(COUNT(*)) OVER(), 2)    AS recommendation_pct,
        ROUND(AVG(final_score), 4)          AS avg_final_score,
        ROUND(AVG(als_normalized), 4)       AS avg_als_score,
        ROUND(AVG(pagerank_score), 4)       AS avg_pagerank_score,
        COUNT(DISTINCT parent_asin)         AS unique_products,
        COUNT(DISTINCT user_index)          AS unique_users
    FROM enriched_recs
    GROUP BY brand
    ORDER BY recommendation_count DESC
    LIMIT 50
""")

log("Top 15 brands in recommendations:")
log(f"{'brand':<30} {'recs':>10} {'pct':>8} {'score':>10}")
log("-" * 60)
for row in brand_dist.limit(15).collect():
    brand = str(row["brand"])[:29]
    log(f"{brand:<30} {row['recommendation_count']:>10,} "
        f"{row['recommendation_pct']:>7}% {row['avg_final_score']:>10}")

save_csv(brand_dist, "brand_recommendation_analytics")

# ─────────────────────────────────────────
# ANALYTICS 3 — SUBCATEGORY DISTRIBUTION
# ─────────────────────────────────────────
section("ANALYTICS 3 — SUBCATEGORY DISTRIBUTION")

subcat_dist = spark.sql("""
    SELECT
        subcategory,
        COUNT(*)                            AS recommendation_count,
        ROUND(COUNT(*) * 100.0
              / SUM(COUNT(*)) OVER(), 2)    AS recommendation_pct,
        ROUND(AVG(final_score), 4)          AS avg_final_score,
        ROUND(AVG(pagerank_score), 4)       AS avg_pagerank_score,
        COUNT(DISTINCT parent_asin)         AS unique_products,
        COUNT(DISTINCT user_index)          AS unique_users
    FROM enriched_recs
    GROUP BY subcategory
    ORDER BY recommendation_count DESC
    LIMIT 50
""")

log("Top 15 subcategories in recommendations:")
log(f"{'subcategory':<35} {'recs':>10} {'pct':>8} {'score':>10}")
log("-" * 65)
for row in subcat_dist.limit(15).collect():
    sub = str(row["subcategory"])[:34]
    log(f"{sub:<35} {row['recommendation_count']:>10,} "
        f"{row['recommendation_pct']:>7}% {row['avg_final_score']:>10}")

save_csv(subcat_dist, "subcategory_recommendation_analytics")

# ─────────────────────────────────────────
# ANALYTICS 4 — SCORE DISTRIBUTION
# ─────────────────────────────────────────
section("ANALYTICS 4 — SCORE DISTRIBUTION")

score_dist = spark.sql("""
    SELECT
        CASE
            WHEN final_score < 0.2 THEN '0.0 - 0.2'
            WHEN final_score < 0.4 THEN '0.2 - 0.4'
            WHEN final_score < 0.6 THEN '0.4 - 0.6'
            WHEN final_score < 0.8 THEN '0.6 - 0.8'
            ELSE                        '0.8 - 1.0'
        END                             AS score_bucket,
        COUNT(*)                        AS count,
        ROUND(COUNT(*) * 100.0
              / SUM(COUNT(*)) OVER(), 2) AS percentage
    FROM enriched_recs
    GROUP BY score_bucket
    ORDER BY score_bucket
""")

log("Final score distribution:")
log(f"{'Score Range':<15} {'Count':>12} {'Percentage':>12}")
log("-" * 40)
for row in score_dist.collect():
    log(f"{row['score_bucket']:<15} {row['count']:>12,} "
        f"{row['percentage']:>11}%")

save_csv(score_dist, "score_distribution")

# ─────────────────────────────────────────
# ANALYTICS 5 — PRICE DISTRIBUTION
# ─────────────────────────────────────────
section("ANALYTICS 5 — PRICE DISTRIBUTION")

price_dist = spark.sql("""
    SELECT
        CASE
            WHEN price IS NULL  THEN 'Unknown'
            WHEN price < 10     THEN 'Under $10'
            WHEN price < 50     THEN '$10 - $49'
            WHEN price < 100    THEN '$50 - $99'
            WHEN price < 500    THEN '$100 - $499'
            ELSE                     '$500+'
        END                             AS price_range,
        COUNT(*)                        AS recommendation_count,
        ROUND(COUNT(*) * 100.0
              / SUM(COUNT(*)) OVER(), 2) AS recommendation_pct,
        ROUND(AVG(final_score), 4)      AS avg_final_score
    FROM enriched_recs
    GROUP BY price_range
    ORDER BY recommendation_count DESC
""")

log("Price distribution in recommendations:")
log(f"{'Price Range':<15} {'Count':>12} {'Pct':>8} {'Score':>10}")
log("-" * 47)
for row in price_dist.collect():
    log(f"{str(row['price_range']):<15} "
        f"{row['recommendation_count']:>12,} "
        f"{row['recommendation_pct']:>7}% "
        f"{row['avg_final_score']:>10}")

save_csv(price_dist, "price_recommendation_analytics")

# ─────────────────────────────────────────
# ANALYTICS 6 — ALS vs PAGERANK CONTRIBUTION
# ─────────────────────────────────────────
section("ANALYTICS 6 — ALS vs PAGERANK CONTRIBUTION")

contribution = spark.sql("""
    SELECT
        ROUND(AVG(als_normalized), 4)           AS avg_als_score,
        ROUND(AVG(pagerank_score), 4)           AS avg_pagerank_score,
        ROUND(AVG(final_score), 4)              AS avg_final_score,
        ROUND(CORR(als_normalized,
                   pagerank_score), 4)          AS als_pagerank_correlation,
        0.7                                     AS als_weight,
        0.3                                     AS pagerank_weight
    FROM enriched_recs
""")

row = contribution.collect()[0]
log(f"Avg ALS score           : {row['avg_als_score']}")
log(f"Avg PageRank score      : {row['avg_pagerank_score']}")
log(f"Avg final score         : {row['avg_final_score']}")
log(f"ALS-PageRank correlation: {row['als_pagerank_correlation']}")

if abs(row["als_pagerank_correlation"]) < 0.3:
    log("Low correlation — ALS and PageRank capture different signals.")
    log("Combining them adds genuine value.")
elif abs(row["als_pagerank_correlation"]) < 0.6:
    log("Moderate correlation — complementary signals.")
else:
    log("High correlation — ALS and PageRank agree strongly.")

save_csv(contribution, "score_contribution_analytics")

# ─────────────────────────────────────────
# ANALYTICS 7 — CATEGORY ANALYTICS
# ─────────────────────────────────────────
section("ANALYTICS 7 — MAIN CATEGORY ANALYTICS")

cat_analytics = spark.sql("""
    SELECT
        main_category,
        COUNT(*)                            AS recommendation_count,
        ROUND(COUNT(*) * 100.0
              / SUM(COUNT(*)) OVER(), 2)    AS recommendation_pct,
        ROUND(AVG(final_score), 4)          AS avg_final_score,
        ROUND(AVG(average_rating), 2)       AS avg_product_rating,
        COUNT(DISTINCT parent_asin)         AS unique_products,
        COUNT(DISTINCT user_index)          AS unique_users
    FROM enriched_recs
    GROUP BY main_category
    ORDER BY recommendation_count DESC
""")

log("Category distribution in recommendations:")
log(f"{'category':<35} {'recs':>10} {'pct':>8} {'score':>10}")
log("-" * 65)
for row in cat_analytics.collect():
    cat = str(row["main_category"])[:34]
    log(f"{cat:<35} {row['recommendation_count']:>10,} "
        f"{row['recommendation_pct']:>7}% "
        f"{row['avg_final_score']:>10}")

save_csv(cat_analytics, "category_recommendation_analytics")

# ─────────────────────────────────────────
# SAVE REPORT
# ─────────────────────────────────────────
section("SAVING REPORT")

spark.createDataFrame(
    [(l,) for l in lines], ["line"]
).coalesce(1).write.mode("overwrite").text(
    OUTPUT_BASE + "/report"
)

log()
log("Analytics CSVs saved:")
log("  top_products_analytics")
log("  brand_recommendation_analytics")
log("  subcategory_recommendation_analytics")
log("  score_distribution")
log("  price_recommendation_analytics")
log("  score_contribution_analytics")
log("  category_recommendation_analytics")
log()
log("=" * 60)
log("RECOMMENDATION SCORING COMPLETE")
log("=" * 60)

spark.stop()