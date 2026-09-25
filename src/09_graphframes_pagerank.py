import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType

# ─────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────
if len(sys.argv) != 4:
    print("Usage: spark-submit 09_graphframes_pagerank.py <sentiment_path> <metadata_path> <output_path>")
    print("Example:")
    print("  spark-submit 09_graphframes_pagerank.py \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/sentiment/appliances/sentiment_data \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/cleaning/appliances/metadata/cleaned_metadata \\")
    print("    hdfs:///user/ss21176_nyu_edu/ecommerce/graph/appliances")
    sys.exit(1)

SENTIMENT_PATH = sys.argv[1]
METADATA_PATH  = sys.argv[2]
OUTPUT_BASE    = sys.argv[3]

# ─────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("GraphFrames_PageRank") \
    .config("spark.sql.adaptive.enabled",      "true") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .config("spark.driver.memory",             "4g") \
    .config("spark.executor.memory",           "4g") \
    .config("spark.sql.shuffle.partitions",    "200") \
    .config("spark.graphx.pregel.checkpointInterval", "2") \
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
# LOAD SENTIMENT DATA
# ─────────────────────────────────────────
section("STEP 1 — LOADING DATA")

df_sentiment = spark.read.parquet(SENTIMENT_PATH)
df_metadata  = spark.read.parquet(METADATA_PATH)

df_sentiment.cache()

total_reviews  = df_sentiment.count()
total_products = df_metadata.count()

log(f"Reviews loaded   : {total_reviews:,}")
log(f"Products loaded  : {total_products:,}")

# ─────────────────────────────────────────
# BUILD CO-PURCHASE GRAPH
# For each user who reviewed 2+ products
# create edges between all pairs of products
# they reviewed — these are implicit co-purchases
# ─────────────────────────────────────────
section("STEP 2 — BUILDING CO-PURCHASE GRAPH")

log("Finding users who reviewed multiple products...")

user_products = df_sentiment.select(
    "user_id", "parent_asin"
).distinct()

users_multi = user_products.groupBy("user_id") \
    .count() \
    .filter(F.col("count") > 1) \
    .select("user_id")

multi_count = users_multi.count()
log(f"Users with 2+ products : {multi_count:,}")

log("Creating product pairs from co-reviews...")

df_multi = user_products.join(users_multi, on="user_id", how="inner")

df_pairs = df_multi.alias("a").join(
    df_multi.alias("b"),
    on="user_id",
    how="inner"
).filter(
    F.col("a.parent_asin") < F.col("b.parent_asin")
).select(
    F.col("a.parent_asin").alias("src"),
    F.col("b.parent_asin").alias("dst")
)

edge_count = df_pairs.count()
log(f"Total co-purchase edges : {edge_count:,}")

log("Aggregating edge weights...")
edges = df_pairs.groupBy("src", "dst") \
    .count() \
    .withColumnRenamed("count", "weight")

edges.cache()
weighted_edge_count = edges.count()
log(f"Unique weighted edges   : {weighted_edge_count:,}")

log()
log("Top 10 strongest co-purchase pairs:")
log(f"{'src':<15} {'dst':<15} {'weight':>10}")
log("-" * 42)
for row in edges.orderBy(F.col("weight").desc()).limit(10).collect():
    log(f"{row['src']:<15} {row['dst']:<15} {row['weight']:>10,}")

# ─────────────────────────────────────────
# BUILD VERTICES
# Every unique product is a vertex
# ─────────────────────────────────────────
section("STEP 3 — BUILDING GRAPH VERTICES")

vertices = df_sentiment.select(
    F.col("parent_asin").alias("id")
).distinct()

vertex_count = vertices.count()
log(f"Total vertices (products) : {vertex_count:,}")

# ─────────────────────────────────────────
# CREATE GRAPHFRAME
# ─────────────────────────────────────────
section("STEP 4 — CREATING GRAPHFRAME")

from graphframes import GraphFrame

g = GraphFrame(vertices, edges)

log(f"GraphFrame created.")
log(f"  Vertices : {g.vertices.count():,}")
log(f"  Edges    : {g.edges.count():,}")

# ─────────────────────────────────────────
# RUN PAGERANK
# resetProbability = 0.15 is standard
# maxIter = 10 is enough for convergence
# ─────────────────────────────────────────
section("STEP 5 — RUNNING PAGERANK")

log("Running PageRank...")
log("  resetProbability : 0.15 (standard damping factor)")
log("  maxIter          : 10")

spark.sparkContext.setCheckpointDir(
    "hdfs:///user/ss21176_nyu_edu/ecommerce/graph/checkpoints"
)

pagerank_result = g.pageRank(
    resetProbability=0.15,
    maxIter=10
)

pr_vertices = pagerank_result.vertices \
    .withColumnRenamed("pagerank", "pagerank_score")

pr_vertices.cache()
pr_count = pr_vertices.count()

log(f"PageRank computed for {pr_count:,} products.")

pr_stats = pr_vertices.select(
    F.min("pagerank_score").alias("min_pr"),
    F.max("pagerank_score").alias("max_pr"),
    F.round(F.avg("pagerank_score"), 6).alias("avg_pr"),
    F.round(F.stddev("pagerank_score"), 6).alias("stddev_pr")
).collect()[0]

log()
log(f"PageRank score statistics:")
log(f"  Min    : {pr_stats['min_pr']}")
log(f"  Max    : {pr_stats['max_pr']}")
log(f"  Avg    : {pr_stats['avg_pr']}")
log(f"  Stddev : {pr_stats['stddev_pr']}")

log()
log("Top 15 products by PageRank score:")
log(f"{'parent_asin':<15} {'pagerank_score':>16}")
log("-" * 33)
for row in pr_vertices.orderBy(
    F.col("pagerank_score").desc()
).limit(15).collect():
    log(f"{row['id']:<15} {row['pagerank_score']:>16.6f}")

# ─────────────────────────────────────────
# NORMALIZE PAGERANK SCORES
# Scale to 0-1 range for combination with ALS
# ─────────────────────────────────────────
section("STEP 6 — NORMALIZING PAGERANK SCORES")

min_pr = pr_stats["min_pr"]
max_pr = pr_stats["max_pr"]

pr_normalized = pr_vertices.withColumn(
    "pagerank_normalized",
    F.round(
        (F.col("pagerank_score") - min_pr) / (max_pr - min_pr),
        6
    )
).withColumnRenamed("id", "parent_asin")

log("PageRank scores normalized to [0, 1] range.")
log("Sample normalized scores:")
log(f"{'parent_asin':<15} {'raw_score':>12} {'normalized':>12}")
log("-" * 42)
for row in pr_normalized.orderBy(
    F.col("pagerank_normalized").desc()
).limit(10).collect():
    log(f"{row['parent_asin']:<15} {row['pagerank_score']:>12.6f} {row['pagerank_normalized']:>12.6f}")

# ─────────────────────────────────────────
# LABEL PROPAGATION — COMMUNITY DETECTION
# ─────────────────────────────────────────
section("STEP 7 — COMMUNITY DETECTION (Label Propagation)")

log("Running label propagation for community detection...")
log("  maxIter : 5")

communities = g.labelPropagation(maxIter=5)
communities.cache()

community_count = communities.select("label").distinct().count()
log(f"Communities found : {community_count:,}")

log()
log("Community size distribution (top 15):")
log(f"{'community_label':<20} {'product_count':>15}")
log("-" * 37)
for row in communities.groupBy("label") \
    .count() \
    .orderBy(F.col("count").desc()) \
    .limit(15).collect():
    log(f"{str(row['label']):<20} {row['count']:>15,}")

# ─────────────────────────────────────────
# JOIN PAGERANK WITH METADATA
# ─────────────────────────────────────────
section("STEP 8 — JOINING PAGERANK WITH METADATA")

pr_with_meta = pr_normalized.join(
    df_metadata.select(
        "parent_asin", "title", "brand",
        "subcategory", "main_category", "price",
        "average_rating", "rating_number", "image_url"
    ),
    on="parent_asin",
    how="left"
)

pr_with_meta.cache()
log(f"Products with PageRank + metadata : {pr_with_meta.count():,}")

log()
log("Top 10 products by PageRank with metadata:")
log(f"{'parent_asin':<15} {'title':<40} {'pr_norm':>10} {'brand':<20}")
log("-" * 88)
for row in pr_with_meta.orderBy(
    F.col("pagerank_normalized").desc()
).limit(10).collect():
    title = str(row["title"])[:39]
    brand = str(row["brand"])[:19]
    log(f"{row['parent_asin']:<15} {title:<40} {row['pagerank_normalized']:>10.4f} {brand:<20}")

# ─────────────────────────────────────────
# JOIN COMMUNITIES WITH METADATA
# ─────────────────────────────────────────
section("STEP 9 — JOINING COMMUNITIES WITH METADATA")

communities_with_meta = communities.join(
    df_metadata.select(
        "parent_asin", "title", "subcategory", "main_category"
    ),
    communities.id == df_metadata.parent_asin,
    how="left"
).select(
    communities.id.alias("parent_asin"),
    "label",
    "title",
    "subcategory",
    "main_category"
)

log("Comparing communities with categories:")
log("(communities should roughly align with product categories)")
log()

community_category = communities_with_meta.groupBy(
    "label", "main_category"
).count() \
 .orderBy("label", F.col("count").desc())

log("Top community-category pairs:")
log(f"{'label':<20} {'main_category':<35} {'count':>8}")
log("-" * 65)
for row in community_category.limit(20).collect():
    cat = str(row["main_category"])[:34]
    log(f"{str(row['label']):<20} {cat:<35} {row['count']:>8,}")

# ─────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────
section("STEP 10 — SAVING OUTPUTS")

pr_normalized.write.mode("overwrite").parquet(
    OUTPUT_BASE + "/pagerank_scores"
)
log(f"PageRank scores saved  : {OUTPUT_BASE}/pagerank_scores")

pr_with_meta.write.mode("overwrite").parquet(
    OUTPUT_BASE + "/pagerank_with_metadata"
)
log(f"PageRank + metadata    : {OUTPUT_BASE}/pagerank_with_metadata")

communities.write.mode("overwrite").parquet(
    OUTPUT_BASE + "/communities"
)
log(f"Communities saved      : {OUTPUT_BASE}/communities")

pr_with_meta.orderBy(F.col("pagerank_normalized").desc()) \
    .coalesce(1) \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv(OUTPUT_BASE + "/pagerank_tableau")
log(f"Tableau CSV saved      : {OUTPUT_BASE}/pagerank_tableau")

summary_data = [
    ("total_reviews",          str(total_reviews)),
    ("total_products",         str(total_products)),
    ("users_with_multi_prods", str(multi_count)),
    ("total_edges",            str(edge_count)),
    ("unique_weighted_edges",  str(weighted_edge_count)),
    ("total_vertices",         str(vertex_count)),
    ("pagerank_min",           str(pr_stats["min_pr"])),
    ("pagerank_max",           str(pr_stats["max_pr"])),
    ("pagerank_avg",           str(pr_stats["avg_pr"])),
    ("communities_found",      str(community_count)),
]

summary_df = spark.createDataFrame(summary_data, ["metric", "value"])
summary_df.coalesce(1).write.mode("overwrite") \
    .option("header", "true").csv(OUTPUT_BASE + "/summary")

report_df = spark.createDataFrame([(l,) for l in lines], ["line"])
report_df.coalesce(1).write.mode("overwrite").text(OUTPUT_BASE + "/report")

log()
log("Summary CSV saved.")
log("Full report saved.")
log()
log("=" * 60)
log("GRAPHFRAMES PAGERANK COMPLETE")
log("=" * 60)

spark.stop()