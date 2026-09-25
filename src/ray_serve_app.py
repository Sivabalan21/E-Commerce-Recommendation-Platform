import ray
from ray import serve
import numpy as np
import pandas as pd
import json
import os

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/content/artifacts")

@serve.deployment(
    name="ALSScorer",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5}
)
class ALSScorer:

    def __init__(self):
        print("ALSScorer: Loading artifacts...")
        self.user_factors = np.load(f"{ARTIFACTS_DIR}/user_factors.npy")
        self.item_factors = np.load(f"{ARTIFACTS_DIR}/item_factors.npy")
        with open(f"{ARTIFACTS_DIR}/user_id_to_index.json") as f:
            self.user_id_to_index = json.load(f)
        self.n_users    = self.user_factors.shape[0]
        self.n_products = self.item_factors.shape[0]
        print(f"ALSScorer: Loaded {self.n_users:,} users and {self.n_products:,} products.")

    async def score(self, user_id: str, top_k: int = 50):
        if user_id not in self.user_id_to_index:
            return None, False
        user_idx    = self.user_id_to_index[user_id]
        user_vector = self.user_factors[user_idx]
        scores      = self.item_factors @ user_vector
        als_norm    = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        top_indices = np.argsort(als_norm)[-top_k:][::-1]
        top_scores  = als_norm[top_indices]
        return list(zip(top_indices.tolist(), top_scores.tolist())), True

    async def score_batch(self, user_indices: list, top_k: int = 10):
        user_vecs   = self.user_factors[user_indices]
        scores      = user_vecs @ self.item_factors.T
        min_s       = scores.min(axis=1, keepdims=True)
        max_s       = scores.max(axis=1, keepdims=True)
        als_norm    = (scores - min_s) / (max_s - min_s + 1e-8)
        top_indices = np.argsort(als_norm, axis=1)[:, -top_k:][:, ::-1]
        top_scores  = np.take_along_axis(als_norm, top_indices, axis=1)
        return top_indices.tolist(), top_scores.tolist()

    async def random_user_indices(self, n: int):
        return np.random.choice(self.n_users, size=n, replace=False).tolist()

    async def __call__(self, request):
        return {"status": "ALSScorer ready"}


@serve.deployment(
    name="PageRankReranker",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5}
)
class PageRankReranker:

    def __init__(self, als_scorer):
        print("PageRankReranker: Loading artifacts...")
        self.als_scorer = als_scorer

        with open(f"{ARTIFACTS_DIR}/index_to_asin.json") as f:
            self.index_to_asin = json.load(f)

        self.pagerank = pd.read_csv(f"{ARTIFACTS_DIR}/pagerank.csv").set_index("parent_asin")
        self.metadata = pd.read_csv(f"{ARTIFACTS_DIR}/metadata.csv").set_index("parent_asin")

        n_products          = len(self.index_to_asin)
        self.pagerank_array = np.zeros(n_products, dtype=np.float32)
        for idx_str, asin in self.index_to_asin.items():
            idx = int(idx_str)
            if asin in self.pagerank.index:
                self.pagerank_array[idx] = float(
                    self.pagerank.loc[asin, "pagerank_normalized"]
                )
        print(f"PageRankReranker: Ready. {len(self.index_to_asin):,} products indexed.")

    def _enrich(self, asin, als_score, pr_score, final_score, rank):
        import math
        def cf(v, d=0.0):
            try:
                x = float(v)
                return d if math.isnan(x) or math.isinf(x) else round(x, 4)
            except: return d
        def cs(v, d=""):
            try:
                s = str(v)
                return d if s in ("nan", "None", "NaN", "") else s
            except: return d
        try:
            meta = self.metadata.loc[asin]
            return {
                "rank"           : rank,
                "parent_asin"    : asin,
                "title"          : cs(meta.get("title")),
                "brand"          : cs(meta.get("brand")),
                "subcategory"    : cs(meta.get("subcategory")),
                "main_category"  : cs(meta.get("main_category")),
                "price"          : cf(meta.get("price")),
                "image_url"      : cs(meta.get("image_url")),
                "average_rating" : cf(meta.get("average_rating")),
                "sentiment_label": cs(meta.get("sentiment_label")),
                "als_score"      : round(float(als_score), 4),
                "pagerank_score" : round(float(pr_score), 4),
                "final_score"    : round(float(final_score), 4),
            }
        except Exception:
            return None

    async def rerank(self, candidates: list, top_k: int = 10):
        results = []
        for idx, als_score in candidates:
            pr_score = float(self.pagerank_array[idx])
            final    = 0.7 * als_score + 0.3 * pr_score
            asin     = self.index_to_asin.get(str(idx))
            if asin:
                results.append((asin, als_score, pr_score, final))
        results.sort(key=lambda x: x[3], reverse=True)
        recommendations = []
        for rank, (asin, als, pr, final) in enumerate(results[:top_k], 1):
            enriched = self._enrich(asin, als, pr, final, rank)
            if enriched:
                recommendations.append(enriched)
        return recommendations

    async def rerank_batch(self, top_indices: list, top_scores: list, top_k: int = 10):
        all_recommendations = []
        for user_top_indices, user_top_scores in zip(top_indices, top_scores):
            candidates = list(zip(user_top_indices, user_top_scores))
            recs = await self.rerank(candidates, top_k)
            all_recommendations.append(recs)
        return all_recommendations

    async def get_popular_by_pagerank(self, top_k: int = 10):
        top_indices = np.argsort(self.pagerank_array)[-top_k:][::-1]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            asin = self.index_to_asin.get(str(idx))
            if asin:
                pr_score = float(self.pagerank_array[idx])
                enriched = self._enrich(asin, 0.0, pr_score, pr_score, rank)
                if enriched:
                    enriched["cold_start"] = True
                    results.append(enriched)
        return results

    async def aggregate_analytics(self, all_asins: list):
        import math
        def cf(v, d=0.0):
            try:
                x = float(v)
                return d if math.isnan(x) or math.isinf(x) else round(x, 2)
            except: return d
        def cs(v, d=""):
            try:
                s = str(v)
                return d if s in ("nan", "None", "NaN") else s
            except: return d

        asin_series  = pd.Series(all_asins)
        unique_asins = asin_series.unique()
        meta_subset  = self.metadata[self.metadata.index.isin(unique_asins)]

        top_products_counts = asin_series.value_counts().head(20)
        top_products = []
        for asin, count in top_products_counts.items():
            try:
                meta = self.metadata.loc[asin]
                top_products.append({
                    "parent_asin"          : asin,
                    "title"                : cs(meta.get("title")),
                    "brand"                : cs(meta.get("brand")),
                    "subcategory"          : cs(meta.get("subcategory")),
                    "price"                : cf(meta.get("price")),
                    "image_url"            : cs(meta.get("image_url")),
                    "recommendation_count" : int(count),
                })
            except Exception:
                continue

        brand_dist = meta_subset["brand"].value_counts().head(10).to_dict() \
            if "brand" in meta_subset.columns else {}
        cat_dist   = meta_subset["subcategory"].value_counts().head(10).to_dict() \
            if "subcategory" in meta_subset.columns else {}
        sent_dist  = meta_subset["sentiment_label"].value_counts().to_dict() \
            if "sentiment_label" in meta_subset.columns else {}

        price_stats = {}
        if "price" in meta_subset.columns:
            prices = meta_subset["price"].dropna()
            if len(prices) > 0:
                price_stats = {
                    "avg_price": round(float(prices.mean()), 2),
                    "min_price": round(float(prices.min()), 2),
                    "max_price": round(float(prices.max()), 2),
                }

        return {
            "top_products"          : top_products,
            "brand_distribution"    : brand_dist,
            "category_distribution" : cat_dist,
            "sentiment_distribution": sent_dist,
            "price_statistics"      : price_stats,
            "unique_products"       : int(len(unique_asins)),
        }

    async def __call__(self, request):
        return {"status": "PageRankReranker ready"}


als_scorer = ALSScorer.bind()
reranker   = PageRankReranker.bind(als_scorer)