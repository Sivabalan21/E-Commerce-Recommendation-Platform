from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse

API_KEY = "smartcart-2026"

@serve.deployment(
    name="API",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5}
)
class APIDeployment:

    def __init__(self, als_scorer, reranker):
        self.als_scorer = als_scorer
        self.reranker   = reranker
        print("API: Ready.")

    async def __call__(self, request: Request):
        path    = request.url.path
        method  = request.method
        api_key = request.headers.get("x-api-key", "")

        headers = {
            "Access-Control-Allow-Origin" : "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*",
        }

        if method == "OPTIONS":
            return JSONResponse({}, headers=headers)

        # Serve UI
        if path in ("/", "/index.html") and method == "GET":
            with open("/content/index.html", "r") as f:
                html = f.read()
            return HTMLResponse(html)

        # Health check
        if path == "/health":
            return JSONResponse({
                "status"  : "healthy",
                "model"   : "ALS rank=50 regParam=0.1 maxIter=20",
                "pipeline": "ALSScorer -> PageRankReranker",
                "version" : "2.0.0"
            }, headers=headers)

        # Auth check
        if api_key != API_KEY:
            return JSONResponse(
                {"error": "Unauthorized"}, status_code=401, headers=headers
            )

        # Recommend
        if path.startswith("/recommend/") and method == "GET":
            user_id = path.split("/recommend/")[1].split("?")[0]
            top_k   = int(request.query_params.get("top_k", 10))
            candidates, found = await self.als_scorer.score.remote(
                user_id=user_id, top_k=50
            )
            if not found:
                popular = await self.reranker.get_popular_by_pagerank.remote(
                    top_k=top_k
                )
                return JSONResponse({
                    "user_id"        : user_id,
                    "cold_start"     : True,
                    "message"        : "User not in training data. Showing popular products.",
                    "recommendations": popular
                }, headers=headers)
            recs = await self.reranker.rerank.remote(
                candidates=candidates, top_k=top_k
            )
            return JSONResponse({
                "user_id"        : user_id,
                "cold_start"     : False,
                "recommendations": recs
            }, headers=headers)

        # Simulate
        if path.startswith("/simulate") and method == "POST":
            n_users = int(request.query_params.get("n_users", 10000))
            n_users = min(max(n_users, 100), 10000)
            user_indices = await self.als_scorer.random_user_indices.remote(
                n=n_users
            )
            top_indices, top_scores = await self.als_scorer.score_batch.remote(
                user_indices=user_indices, top_k=10
            )
            all_recs = await self.reranker.rerank_batch.remote(
                top_indices=top_indices,
                top_scores=top_scores,
                top_k=10
            )
            all_asins = [
                rec["parent_asin"]
                for user_recs in all_recs
                for rec in user_recs
            ]
            analytics = await self.reranker.aggregate_analytics.remote(
                all_asins=all_asins
            )
            return JSONResponse({
                "users_simulated": n_users,
                **analytics
            }, headers=headers)

        return JSONResponse(
            {"error": f"Not found: {method} {path}"},
            status_code=404,
            headers=headers
        )