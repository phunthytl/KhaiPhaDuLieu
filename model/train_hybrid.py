"""
Gán trọng số hybrid thủ công theo 3 tier (CF / Content / Demographic),
tính dự đoán trên tập test và đánh giá RMSE/MAE + ranking theo top-10 phim / user.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Trọng số thủ công (tổng mỗi tier = 1.0): CF, Content, Demographic ---
HYBRID_WEIGHTS_BY_TIER = {
    "Tier1_New": {"CF": 0.25, "Content": 0.35, "Demographic": 0.40},
    "Tier2_Medium": {"CF": 0.45, "Content": 0.30, "Demographic": 0.25},
    "Tier3_Old": {"CF": 0.70, "Content": 0.15, "Demographic": 0.15},
}
DEFAULT_WEIGHTS = {"CF": 0.34, "Content": 0.33, "Demographic": 0.33}

# --- Đánh giá ---


def calculate_errors(true_ratings, pred_ratings):
    rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
    mae = mean_absolute_error(true_ratings, pred_ratings)
    return rmse, mae


def ranking_topk_per_user(
    test_df: pd.DataFrame,
    k: int = 10,
    min_items_per_user: int = 10,
    relevance_min_rating: float = 4.0,
):
    """
    Với mỗi user có ít nhất `min_items_per_user` dòng trong tập test:
    sắp xếp các phim (test) theo predicted_rating giảm dần, lấy top `k`,
    rồi tính Precision/Recall/NDCG (relevant = rating >= relevance_min_rating).

    Recall theo user = (số relevant trong top-k) / (tổng relevant của user trong toàn bộ test của user).
    NDCG: relevance nhị phân; IDCG = sắp tối đa k slot với các phim relevant.
    """
    precisions, recalls, ndcgs = [], [], []
    rating_col, pred_col = "rating", "predicted_rating"

    for _, g in test_df.groupby("user_id", sort=False):
        if len(g) < min_items_per_user:
            continue
        n_rel = int((g[rating_col] >= relevance_min_rating).sum())
        if n_rel < 1:
            continue

        top = g.nlargest(k, pred_col)
        hits = int((top[rating_col] >= relevance_min_rating).sum())
        precisions.append(hits / float(k))
        recalls.append(hits / float(n_rel))

        dcg = 0.0
        for i, (_, row) in enumerate(top.reset_index(drop=True).iterrows()):
            rel = 1.0 if row[rating_col] >= relevance_min_rating else 0.0
            dcg += rel / np.log2(i + 2)

        r_cap = min(k, n_rel)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(r_cap))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    if not precisions:
        return {
            "k": k,
            "min_test_ratings_per_user": min_items_per_user,
            "relevance_rating_min": relevance_min_rating,
            "users_evaluated": 0,
            "precision_avg": 0.0,
            "recall_avg": 0.0,
            "ndcg_avg": 0.0,
        }

    return {
        "k": k,
        "min_test_ratings_per_user": min_items_per_user,
        "relevance_rating_min": relevance_min_rating,
        "users_evaluated": len(precisions),
        "precision_avg": float(np.mean(precisions)),
        "recall_avg": float(np.mean(recalls)),
        "ndcg_avg": float(np.mean(ndcgs)),
    }


def evaluate_model(test_set: pd.DataFrame):
    rmse, mae = calculate_errors(test_set["rating"], test_set["predicted_rating"])
    ranking = ranking_topk_per_user(test_set, k=10, min_items_per_user=10)
    return rmse, mae, ranking


# --- Pipeline ---

print("=" * 70)
print("TRỌNG SỐ HYBRID THỦ CÔNG + ĐÁNH GIÁ (TOP-10 / USER TRÊN TẬP TEST)")
print("=" * 70)

print("\n[1] Đang tải dữ liệu và mô hình...")
final_dataset = pd.read_csv("data/final_dataset.csv")
user_segments = pd.read_csv("data/user_segments.csv")

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_model_dir = os.path.join(_root, "model")

with open(os.path.join(_model_dir, "cf_model.pkl"), "rb") as f:
    cf_model = pickle.load(f)
with open(os.path.join(_model_dir, "content_model.pkl"), "rb") as f:
    content_model = pickle.load(f)
with open(os.path.join(_model_dir, "demographic_model.pkl"), "rb") as f:
    demographic_model = pickle.load(f)

content_scores = content_model.get("content_scores", {})
demographic_scores = demographic_model.get("demographic_scores", {})

cf_user_list = cf_model.get("user_list", [])
cf_movie_list = cf_model.get("movie_list", [])
cf_matrix = cf_model.get("train_matrix", np.array([]))


def get_cf_score(user_id, movie_id):
    if user_id not in cf_user_list or movie_id not in cf_movie_list:
        return 0.5
    user_idx = cf_user_list.index(user_id)
    movie_idx = cf_movie_list.index(movie_id)
    try:
        score = cf_matrix[user_idx, movie_idx]
        return np.clip(score / 5.0 if score > 0 else 0.5, 0, 1)
    except Exception:
        return 0.5


print("\n[2] Phân chia tập train/test...")

final_dataset = final_dataset.merge(
    user_segments[["user_id", "tier"]],
    on="user_id",
    how="left",
)

np.random.seed(42)
test_mask = np.random.rand(len(final_dataset)) < 0.2
train_set = final_dataset[~test_mask].copy()
test_set = final_dataset[test_mask].copy()

hybrid_weights = {t: dict(w) for t, w in HYBRID_WEIGHTS_BY_TIER.items()}
for t in ["Tier1_New", "Tier2_Medium", "Tier3_Old"]:
    if t not in hybrid_weights:
        hybrid_weights[t] = dict(DEFAULT_WEIGHTS)

print("\n[3] Trọng số hybrid (cố định theo tier):")
for tier, w in hybrid_weights.items():
    print(
        f"  {tier}: CF={w['CF']}, Content={w['Content']}, Demographic={w['Demographic']}"
    )

print("\n[4] Tính điểm dự đoán hybrid trên tập test...")


def get_hybrid_score(user_id, movie_id, tier):
    cf_score = get_cf_score(user_id, movie_id)
    content_score = content_scores.get(user_id, {}).get(movie_id, 0.5)
    demo_score = demographic_scores.get(user_id, {}).get(movie_id, 0.5)
    weights = hybrid_weights.get(tier, DEFAULT_WEIGHTS)
    hybrid = (
        cf_score * weights["CF"]
        + content_score * weights["Content"]
        + demo_score * weights["Demographic"]
    )
    return np.clip(hybrid, 0, 1)


def predict_batch(df: pd.DataFrame) -> np.ndarray:
    tiers = df["tier"].fillna("Tier2_Medium").astype(str).to_numpy()
    u = df["user_id"].to_numpy(dtype=np.int64)
    m = df["movie_id"].to_numpy(dtype=np.int64)
    out = np.zeros(len(df), dtype=np.float64)
    for i in range(len(df)):
        w = hybrid_weights.get(tiers[i], DEFAULT_WEIGHTS)
        uid, mid = int(u[i]), int(m[i])
        cf_s = get_cf_score(uid, mid)
        c_s = content_scores.get(uid, {}).get(mid, 0.5)
        d_s = demographic_scores.get(uid, {}).get(mid, 0.5)
        out[i] = np.clip(cf_s * w["CF"] + c_s * w["Content"] + d_s * w["Demographic"], 0, 1)
    return out * 5.0


test_set = test_set.copy()
test_set["predicted_rating"] = predict_batch(test_set)

print("\n[5] Đánh giá (RMSE/MAE + top-10 phim / user trong test)...")

rmse, mae, ranking = evaluate_model(test_set)

print("\n[6] Đang lưu kết quả...")

with open(os.path.join(_model_dir, "hybrid_weights.json"), "w", encoding="utf-8") as f:
    json.dump(hybrid_weights, f, indent=2)

evaluation = {
    "global": {
        "rmse": float(rmse),
        "mae": float(mae),
        "train_size": int(len(train_set)),
        "test_size": int(len(test_set)),
        "train_set_size": int(len(train_set)),
        "test_set_size": int(len(test_set)),
    },
    "ranking_metrics": {
        "method": "top10_movies_per_user_test_pool",
        "description": (
            "Chỉ user có ≥10 rating trong test; trong các phim đó, "
            "lấy top 10 theo predicted_rating; relevant = rating >= 4."
        ),
        "k": ranking["k"],
        "min_test_ratings_per_user": ranking["min_test_ratings_per_user"],
        "relevance_rating_min": ranking["relevance_rating_min"],
        "users_evaluated": ranking["users_evaluated"],
        "precision_at_10": ranking["precision_avg"],
        "recall_at_10": ranking["recall_avg"],
        "ndcg_at_10": ranking["ndcg_avg"],
    },
}

with open(os.path.join(_model_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(evaluation, f, indent=2)

print("\n" + "=" * 70)
print("HOÀN TẤT")
print("=" * 70)
print("\nChỉ số:")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE: {mae:.4f}")
print(
    f"  Top-10/user (users đủ điều kiện: {ranking['users_evaluated']}): "
    f"P@10={ranking['precision_avg']:.4f}, R@10={ranking['recall_avg']:.4f}, "
    f"NDCG@10={ranking['ndcg_avg']:.4f}"
)
print("\nĐã lưu:")
print("  - model/hybrid_weights.json")
print("  - model/evaluation_metrics.json")
print("=" * 70)
