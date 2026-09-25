import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ─────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────
if len(sys.argv) != 3:
    print("Usage: spark-submit 07_spark_sql_analytics.py <input_path> <output_path>")
    print("Example:")
    print("  spark-submit 07_spark_sql_analytics.py \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/sentiment/appliances/sentiment_data \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/analytics/appliances")
    sys.exit(1)

INPUT_PATH  = sys.argv[1]
OUTPUT_BASE = sys.argv[2]

# ─────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("Spark_SQL_Analytics") \
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

def save_csv(df, name):
    path = OUTPUT_BASE + "/" + name
    df.coalesce(1).write.mode("overwrite") \
      .option("header", "true").csv(path)
    log(f"Saved: {name}")
    return path

# ─────────────────────────────────────────
# LOAD SENTIMENT ENRICHED DATA
# ─────────────────────────────────────────
section("STEP 1 — LOADING SENTIMENT ENRICHED DATA")

df = spark.read.parquet(INPUT_PATH)
df.cache()

total_rows = df.count()
log(f"Total rows : {total_rows:,}")
log(f"Columns    : {df.columns}")

df.createOrReplaceTempView("reviews")

# ─────────────────────────────────────────
# ANALYTICS 1 — REVIEW VOLUME BY CATEGORY
# ─────────────────────────────────────────
section("ANALYTICS 1 — REVIEW VOLUME BY CATEGORY")

cat_volume = spark.sql("""
    SELECT
        main_category,
        COUNT(*)                                    AS review_count,
        ROUND(AVG(rating), 2)                       AS avg_rating,
        ROUND(AVG(sentiment_score), 4)              AS avg_sentiment,
        ROUND(SUM(CASE WHEN sentiment_label = 'Positive'
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS positive_pct,
        ROUND(SUM(CASE WHEN sentiment_label = 'Negative'
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS negative_pct,
        ROUND(AVG(helpful_vote), 2)                 AS avg_helpful_votes
    FROM reviews
    GROUP BY main_category
    ORDER BY review_count DESC
""")

log(f"{'main_category':<35} {'reviews':>10} {'avg_rating':>12} {'avg_sentiment':>14} {'positive%':>10}")
log("-" * 84)
for row in cat_volume.collect():
    cat = str(row["main_category"])[:34]
    log(f"{cat:<35} {row['review_count']:>10,} {row['avg_rating']:>12} {row['avg_sentiment']:>14} {row['positive_pct']:>9}%")

save_csv(cat_volume, "category_analytics")

# ─────────────────────────────────────────
# ANALYTICS 2 — BRAND ANALYTICS
# ─────────────────────────────────────────
section("ANALYTICS 2 — BRAND ANALYTICS (top 20)")

brand_analytics = spark.sql("""
    SELECT
        brand,
        COUNT(*)                                    AS review_count,
        ROUND(AVG(rating), 2)                       AS avg_rating,
        ROUND(AVG(sentiment_score), 4)              AS avg_sentiment,
        ROUND(SUM(CASE WHEN sentiment_label = 'Positive'
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS positive_pct,
        ROUND(SUM(CASE WHEN sentiment_label = 'Negative'
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS negative_pct,
        ROUND(AVG(helpful_vote), 2)                 AS avg_helpful_votes,
        COUNT(DISTINCT parent_asin)                 AS product_count,
        ROUND(AVG(price), 2)                        AS avg_price
    FROM reviews
    GROUP BY brand
    HAVING COUNT(*) >= 1000
    ORDER BY review_count DESC
    LIMIT 20
""")

log(f"{'brand':<30} {'reviews':>10} {'avg_rating':>12} {'avg_sentiment':>14} {'products':>10}")
log("-" * 78)
for row in brand_analytics.collect():
    brand = str(row["brand"])[:29]
    log(f"{brand:<30} {row['review_count']:>10,} {row['avg_rating']:>12} {row['avg_sentiment']:>14} {row['product_count']:>10,}")

save_csv(brand_analytics, "brand_analytics")

# ─────────────────────────────────────────
# ANALYTICS 3 — SUBCATEGORY ANALYTICS
# ─────────────────────────────────────────
section("ANALYTICS 3 — SUBCATEGORY ANALYTICS (top 20)")

subcat_analytics = spark.sql("""
    SELECT
        subcategory,
        COUNT(*)                                    AS review_count,
        ROUND(AVG(rating), 2)                       AS avg_rating,
        ROUND(AVG(sentiment_score), 4)              AS avg_sentiment,
        ROUND(SUM(CASE WHEN sentiment_label = 'Positive'
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS positive_pct,
        ROUND(SUM(CASE WHEN sentiment_label = 'Negative'
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS negative_pct,
        COUNT(DISTINCT parent_asin)                 AS product_count,
        ROUND(AVG(price), 2)                        AS avg_price
    FROM reviews
    GROUP BY subcategory
    ORDER BY review_count DESC
    LIMIT 20
""")

log(f"{'subcategory':<35} {'reviews':>10} {'avg_rating':>12} {'avg_sentiment':>14} {'products':>10}")
log("-" * 84)
for row in subcat_analytics.collect():
    sub = str(row["subcategory"])[:34]
    log(f"{sub:<35} {row['review_count']:>10,} {row['avg_rating']:>12} {row['avg_sentiment']:>14} {row['product_count']:>10,}")

save_csv(subcat_analytics, "subcategory_analytics")

# ─────────────────────────────────────────
# ANALYTICS 4 — RATING DISTRIBUTION
# ─────────────────────────────────────────
section("ANALYTICS 4 — RATING DISTRIBUTION")

rating_dist = spark.sql("""
    SELECT
        rating,
        COUNT(*)                                    AS review_count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*))
              OVER(), 2)                            AS percentage,
        ROUND(AVG(sentiment_score), 4)              AS avg_sentiment,
        ROUND(AVG(helpful_vote), 2)                 AS avg_helpful_votes
    FROM reviews
    GROUP BY rating
    ORDER BY rating
""")

log(f"{'Rating':<10} {'Count':>12} {'Percentage':>12} {'Avg Sentiment':>15}")
log("-" * 52)
for row in rating_dist.collect():
    log(f"{row['rating']:<10} {row['review_count']:>12,} {row['percentage']:>11}% {row['avg_sentiment']:>15}")

save_csv(rating_dist, "rating_distribution")

# ─────────────────────────────────────────
# ANALYTICS 5 — SENTIMENT DISAGREEMENT
# High rating but negative sentiment
# Low rating but positive sentiment
# ─────────────────────────────────────────
section("ANALYTICS 5 — SENTIMENT VS RATING DISAGREEMENT")

disagreement = spark.sql("""
    SELECT
        CASE
            WHEN rating >= 4.0 AND sentiment_label = 'Negative'
                THEN 'High rating, Negative sentiment'
            WHEN rating <= 2.0 AND sentiment_label = 'Positive'
                THEN 'Low rating, Positive sentiment'
            WHEN rating >= 4.0 AND sentiment_label = 'Positive'
                THEN 'High rating, Positive sentiment'
            WHEN rating <= 2.0 AND sentiment_label = 'Negative'
                THEN 'Low rating, Negative sentiment'
            ELSE 'Neutral zone'
        END                                         AS disagreement_type,
        COUNT(*)                                    AS review_count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*))
              OVER(), 2)                            AS percentage
    FROM reviews
    GROUP BY disagreement_type
    ORDER BY review_count DESC
""")

log(f"{'Type':<40} {'Count':>12} {'Percentage':>12}")
log("-" * 66)
for row in disagreement.collect():
    log(f"{str(row['disagreement_type']):<40} {row['review_count']:>12,} {row['percentage']:>11}%")

save_csv(disagreement, "sentiment_disagreement")

# ─────────────────────────────────────────
# ANALYTICS 6 — REVIEW TRENDS OVER TIME
# ─────────────────────────────────────────
section("ANALYTICS 6 — REVIEW TRENDS OVER TIME")

time_trends = spark.sql("""
    SELECT
        YEAR(review_date)                           AS year,
        COUNT(*)                                    AS review_count,
        ROUND(AVG(rating), 2)                       AS avg_rating,
        ROUND(AVG(sentiment_score), 4)              AS avg_sentiment,
        ROUND(SUM(CASE WHEN sentiment_label = 'Positive'
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS positive_pct,
        ROUND(SUM(CASE WHEN verified_purchase = true
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS verified_pct
    FROM reviews
    WHERE YEAR(review_date) >= 2010
    GROUP BY year
    ORDER BY year
""")

log(f"{'Year':<8} {'Reviews':>12} {'Avg Rating':>12} {'Avg Sentiment':>15} {'Positive%':>12}")
log("-" * 62)
for row in time_trends.collect():
    log(f"{row['year']:<8} {row['review_count']:>12,} {row['avg_rating']:>12} {row['avg_sentiment']:>15} {row['positive_pct']:>11}%")

save_csv(time_trends, "time_trends")

# ─────────────────────────────────────────
# ANALYTICS 7 — TOP PRODUCTS BY REVIEW COUNT
# ─────────────────────────────────────────
section("ANALYTICS 7 — TOP 50 PRODUCTS BY REVIEW COUNT")

top_products = spark.sql("""
    SELECT
        parent_asin,
        MAX(title)                                  AS title,
        MAX(brand)                                  AS brand,
        MAX(subcategory)                            AS subcategory,
        MAX(price)                                  AS price,
        MAX(image_url)                              AS image_url,
        COUNT(*)                                    AS review_count,
        ROUND(AVG(rating), 2)                       AS avg_rating,
        ROUND(AVG(sentiment_score), 4)              AS avg_sentiment,
        ROUND(SUM(CASE WHEN sentiment_label = 'Positive'
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS positive_pct,
        ROUND(SUM(CASE WHEN sentiment_label = 'Negative'
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS negative_pct,
        SUM(helpful_vote)                           AS total_helpful_votes
    FROM reviews
    GROUP BY parent_asin
    ORDER BY review_count DESC
    LIMIT 50
""")

log(f"{'parent_asin':<15} {'title':<45} {'reviews':>10} {'avg_rating':>12}")
log("-" * 85)
for row in top_products.collect():
    title = str(row["title"])[:44]
    log(f"{row['parent_asin']:<15} {title:<45} {row['review_count']:>10,} {row['avg_rating']:>12}")

save_csv(top_products, "top_products")

# ─────────────────────────────────────────
# ANALYTICS 8 — VERIFIED VS UNVERIFIED
# ─────────────────────────────────────────
section("ANALYTICS 8 — VERIFIED VS UNVERIFIED PURCHASE ANALYSIS")

verified_analysis = spark.sql("""
    SELECT
        verified_purchase,
        COUNT(*)                                    AS review_count,
        ROUND(AVG(rating), 2)                       AS avg_rating,
        ROUND(AVG(sentiment_score), 4)              AS avg_sentiment,
        ROUND(AVG(helpful_vote), 2)                 AS avg_helpful_votes,
        ROUND(SUM(CASE WHEN sentiment_label = 'Positive'
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS positive_pct,
        ROUND(SUM(CASE WHEN sentiment_label = 'Negative'
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS negative_pct
    FROM reviews
    GROUP BY verified_purchase
    ORDER BY verified_purchase DESC
""")

log(f"{'Verified':<12} {'Count':>12} {'Avg Rating':>12} {'Avg Sentiment':>15} {'Positive%':>12}")
log("-" * 66)
for row in verified_analysis.collect():
    log(f"{str(row['verified_purchase']):<12} {row['review_count']:>12,} {row['avg_rating']:>12} {row['avg_sentiment']:>15} {row['positive_pct']:>11}%")

save_csv(verified_analysis, "verified_analysis")

# ─────────────────────────────────────────
# ANALYTICS 9 — PRICE RANGE ANALYSIS
# ─────────────────────────────────────────
section("ANALYTICS 9 — REVIEW PATTERNS BY PRICE RANGE")

price_analysis = spark.sql("""
    SELECT
        CASE
            WHEN price IS NULL          THEN 'Price Unknown'
            WHEN price < 10             THEN 'Under $10'
            WHEN price < 50             THEN '$10 - $49'
            WHEN price < 100            THEN '$50 - $99'
            WHEN price < 500            THEN '$100 - $499'
            ELSE                             '$500+'
        END                                         AS price_range,
        COUNT(*)                                    AS review_count,
        ROUND(AVG(rating), 2)                       AS avg_rating,
        ROUND(AVG(sentiment_score), 4)              AS avg_sentiment,
        ROUND(SUM(CASE WHEN sentiment_label = 'Positive'
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS positive_pct
    FROM reviews
    GROUP BY price_range
    ORDER BY review_count DESC
""")

log(f"{'Price Range':<18} {'Reviews':>12} {'Avg Rating':>12} {'Avg Sentiment':>15} {'Positive%':>12}")
log("-" * 72)
for row in price_analysis.collect():
    log(f"{str(row['price_range']):<18} {row['review_count']:>12,} {row['avg_rating']:>12} {row['avg_sentiment']:>15} {row['positive_pct']:>11}%")

save_csv(price_analysis, "price_analysis")

# ─────────────────────────────────────────
# ANALYTICS 10 — HELPFUL VOTE ANALYSIS
# ─────────────────────────────────────────
section("ANALYTICS 10 — MOST HELPFUL REVIEWS BY CATEGORY")

helpful_analysis = spark.sql("""
    SELECT
        main_category,
        COUNT(*)                                    AS review_count,
        ROUND(AVG(helpful_vote), 2)                 AS avg_helpful_votes,
        MAX(helpful_vote)                           AS max_helpful_votes,
        SUM(helpful_vote)                           AS total_helpful_votes,
        ROUND(SUM(CASE WHEN helpful_vote > 10
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS pct_highly_helpful
    FROM reviews
    GROUP BY main_category
    ORDER BY avg_helpful_votes DESC
""")

log(f"{'main_category':<35} {'avg_helpful':>12} {'max_helpful':>12} {'pct_high':>10}")
log("-" * 72)
for row in helpful_analysis.collect():
    cat = str(row["main_category"])[:34]
    log(f"{cat:<35} {row['avg_helpful_votes']:>12} {row['max_helpful_votes']:>12,} {row['pct_highly_helpful']:>9}%")

save_csv(helpful_analysis, "helpful_analysis")

# ─────────────────────────────────────────
# ANALYTICS 11 — BRAND SENTIMENT RANKING
# Best and worst brands by sentiment
# ─────────────────────────────────────────
section("ANALYTICS 11 — BRAND SENTIMENT RANKING")

brand_sentiment_rank = spark.sql("""
    SELECT
        brand,
        COUNT(*)                                    AS review_count,
        ROUND(AVG(sentiment_score), 4)              AS avg_sentiment,
        ROUND(AVG(rating), 2)                       AS avg_rating,
        ROUND(SUM(CASE WHEN sentiment_label = 'Positive'
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS positive_pct,
        ROUND(SUM(CASE WHEN sentiment_label = 'Negative'
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS negative_pct
    FROM reviews
    WHERE brand != 'Unknown'
    GROUP BY brand
    HAVING COUNT(*) >= 500
    ORDER BY avg_sentiment DESC
""")

rows = brand_sentiment_rank.collect()

log("Top 10 brands by sentiment:")
log(f"{'brand':<30} {'avg_sentiment':>15} {'positive%':>12} {'reviews':>10}")
log("-" * 70)
for row in rows[:10]:
    log(f"{str(row['brand']):<30} {row['avg_sentiment']:>15} {row['positive_pct']:>11}% {row['review_count']:>10,}")

log()
log("Bottom 10 brands by sentiment:")
log(f"{'brand':<30} {'avg_sentiment':>15} {'positive%':>12} {'reviews':>10}")
log("-" * 70)
for row in rows[-10:]:
    log(f"{str(row['brand']):<30} {row['avg_sentiment']:>15} {row['positive_pct']:>11}% {row['review_count']:>10,}")

save_csv(brand_sentiment_rank, "brand_sentiment_ranking")

# ─────────────────────────────────────────
# ANALYTICS 12 — PRODUCT QUALITY SCORE
# Combined metric for Tableau dashboard
# ─────────────────────────────────────────
section("ANALYTICS 12 — PRODUCT QUALITY SCORE (top 100)")

product_quality = spark.sql("""
    SELECT
        parent_asin,
        MAX(title)                                  AS title,
        MAX(brand)                                  AS brand,
        MAX(subcategory)                            AS subcategory,
        MAX(main_category)                          AS main_category,
        MAX(price)                                  AS price,
        MAX(image_url)                              AS image_url,
        COUNT(*)                                    AS review_count,
        ROUND(AVG(rating), 2)                       AS avg_rating,
        ROUND(AVG(sentiment_score), 4)              AS avg_sentiment,
        ROUND(SUM(CASE WHEN sentiment_label = 'Positive'
              THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 2)                        AS positive_pct,
        SUM(helpful_vote)                           AS total_helpful_votes,
        ROUND(
            (AVG(rating) / 5.0 * 0.5) +
            ((AVG(sentiment_score) + 1.0) / 2.0 * 0.3) +
            (LEAST(COUNT(*) / 1000.0, 1.0) * 0.2)
        , 4)                                        AS quality_score
    FROM reviews
    GROUP BY parent_asin
    HAVING COUNT(*) >= 50
    ORDER BY quality_score DESC
    LIMIT 100
""")

log(f"{'parent_asin':<15} {'title':<40} {'quality_score':>14} {'reviews':>10}")
log("-" * 82)
for row in product_quality.collect()[:10]:
    title = str(row["title"])[:39]
    log(f"{row['parent_asin']:<15} {title:<40} {row['quality_score']:>14} {row['review_count']:>10,}")

save_csv(product_quality, "product_quality_scores")

# ─────────────────────────────────────────
# SAVE FULL REPORT
# ─────────────────────────────────────────
section("SAVING REPORT")

report_df = spark.createDataFrame([(l,) for l in lines], ["line"])
report_df.coalesce(1).write.mode("overwrite").text(OUTPUT_BASE + "/report")

log()
log("All analytics CSVs saved.")
log("Full report saved.")
log()
log("=" * 60)
log("SPARK SQL ANALYTICS COMPLETE")
log("=" * 60)
log()
log("Files saved for Tableau:")
log("  category_analytics.csv")
log("  brand_analytics.csv")
log("  subcategory_analytics.csv")
log("  rating_distribution.csv")
log("  sentiment_disagreement.csv")
log("  time_trends.csv")
log("  top_products.csv")
log("  verified_analysis.csv")
log("  price_analysis.csv")
log("  helpful_analysis.csv")
log("  brand_sentiment_ranking.csv")
log("  product_quality_scores.csv")

spark.stop()