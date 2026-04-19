import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Đánh giá (trước đây utils/evaluation.py) ---


def calculate_errors(true_ratings, pred_ratings):
    """RMSE và MAE."""
    rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
    mae = mean_absolute_error(true_ratings, pred_ratings)
    return rmse, mae


def ranking_metrics(true_ratings, pred_ratings, k=5):
    """Precision@K, Recall@K, NDCG@K (relevant: rating >= 4)."""
    ranked = np.argsort(-pred_ratings)[:k]

    relevant = sum(1 for i in ranked if true_ratings[i] >= 4)
    precision_k = relevant / k if k > 0 else 0

    total_relevant = sum(1 for r in true_ratings if r >= 4)
    recall_k = relevant / total_relevant if total_relevant > 0 else 0

    dcg = sum(1.0 / np.log2(i + 2) for i, idx in enumerate(ranked) if true_ratings[idx] >= 4)
    ideal_relevant = sum(1 for r in true_ratings if r >= 4)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(ideal_relevant, k)))
    ndcg_k = dcg / idcg if idcg > 0 else 0

    return {"precision": precision_k, "recall": recall_k, "ndcg": ndcg_k}


def evaluate_model(test_set):
    """RMSE, MAE và ranking @5, @10."""
    rmse, mae = calculate_errors(test_set["rating"], test_set["predicted_rating"])
    y_true = test_set["rating"].values
    y_pred = test_set["predicted_rating"].values
    metrics_5 = ranking_metrics(y_true, y_pred, k=5)
    metrics_10 = ranking_metrics(y_true, y_pred, k=10)
    return rmse, mae, metrics_5, metrics_10


# --- Pipeline hybrid ---

print("=" * 70)
print("HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH LAI (HYBRID) BẰNG MACHINE LEARNING")
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
    """Điểm CF (SVD), chuẩn hóa khoảng [0, 1]."""
    if user_id not in cf_user_list or movie_id not in cf_movie_list:
        return 0.5
    user_idx = cf_user_list.index(user_id)
    movie_idx = cf_movie_list.index(movie_id)
    try:
        score = cf_matrix[user_idx, movie_idx]
        return np.clip(score / 5.0 if score > 0 else 0.5, 0, 1)
    except Exception:
        return 0.5


print("\n[2] Phân chia tập dữ liệu train/test...")

final_dataset = final_dataset.merge(
    user_segments[["user_id", "tier"]],
    on="user_id",
    how="left",
)

np.random.seed(42)
test_mask = np.random.rand(len(final_dataset)) < 0.2
train_set = final_dataset[~test_mask].copy()
test_set = final_dataset[test_mask].copy()

print("\n[3] Học trọng số Hybrid bằng Linear Regression...")

hybrid_weights = {}
tiers = train_set["tier"].unique()
default_weights = {"CF": 0.34, "Content": 0.33, "Demographic": 0.33}

for tier in tiers:
    tier_data = train_set[train_set["tier"] == tier]
    if len(tier_data) == 0:
        continue

    print(f"  Đang huấn luyện Regression cho phân khúc: {tier} ({len(tier_data)} ratings)")

    X = []
    Y = tier_data["rating"].values / 5.0

    for _, row in tier_data.iterrows():
        u_id = row["user_id"]
        m_id = row["movie_id"]
        cf_s = get_cf_score(u_id, m_id)
        cont_s = content_scores.get(u_id, {}).get(m_id, 0.5)
        demo_s = demographic_scores.get(u_id, {}).get(m_id, 0.5)
        X.append([cf_s, cont_s, demo_s])

    X = np.array(X)

    lr = LinearRegression(positive=True, fit_intercept=False)
    lr.fit(X, Y)

    coef = lr.coef_
    if sum(coef) == 0:
        coef = np.array([0.34, 0.33, 0.33])
    coef_norm = coef / sum(coef)

    hybrid_weights[tier] = {
        "CF": round(float(coef_norm[0]), 4),
        "Content": round(float(coef_norm[1]), 4),
        "Demographic": round(float(coef_norm[2]), 4),
    }

    print(
        f"    -> Trọng số học được: CF={hybrid_weights[tier]['CF']}, "
        f"Content={hybrid_weights[tier]['Content']}, "
        f"Demographic={hybrid_weights[tier]['Demographic']}"
    )

for t in ["Tier1_New", "Tier2_Medium", "Tier3_Old"]:
    if t not in hybrid_weights:
        hybrid_weights[t] = default_weights

print("\n[4] Tính toán điểm số hybrid (Dự đoán)...")

test_set["predicted_rating"] = 0.0


def get_hybrid_score(user_id, movie_id, tier):
    cf_score = get_cf_score(user_id, movie_id)
    content_score = content_scores.get(user_id, {}).get(movie_id, 0.5)
    demo_score = demographic_scores.get(user_id, {}).get(movie_id, 0.5)
    weights = hybrid_weights.get(tier, default_weights)
    hybrid = (
        cf_score * weights["CF"]
        + content_score * weights["Content"]
        + demo_score * weights["Demographic"]
    )
    return np.clip(hybrid, 0, 1)


for idx, row in test_set.iterrows():
    score = get_hybrid_score(row["user_id"], row["movie_id"], row["tier"])
    test_set.at[idx, "predicted_rating"] = score * 5.0

print("\n[5] Đang đánh giá mô hình...")

rmse, mae, metrics_5, metrics_10 = evaluate_model(test_set)

print("\n[6] Đang lưu kết quả...")

with open(os.path.join(_model_dir, "hybrid_weights.json"), "w", encoding="utf-8") as f:
    json.dump(hybrid_weights, f, indent=2)

evaluation = {
    "global": {
        "rmse": float(rmse),
        "mae": float(mae),
        "test_size": len(test_set),
        "train_size": len(train_set),
    },
    "ranking_metrics": {
        "precision_5": float(metrics_5["precision"]),
        "recall_5": float(metrics_5["recall"]),
        "ndcg_5": float(metrics_5["ndcg"]),
        "precision_10": float(metrics_10["precision"]),
        "recall_10": float(metrics_10["recall"]),
        "ndcg_10": float(metrics_10["ndcg"]),
    },
}

with open(os.path.join(_model_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(evaluation, f, indent=2)

print("\n" + "=" * 70)
print("HOÀN TẤT HUẤN LUYỆN VÀ ĐÁNH GIÁ (MACHINE LEARNING EDITION)")
print("=" * 70)
print("\nChỉ số độ chính xác (Metrics):")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  Precision@5: {metrics_5['precision']:.4f}")
print(f"  Recall@5: {metrics_5['recall']:.4f}")
print("\nCác file đã được lưu:")
print("  - model/hybrid_weights.json")
print("  - model/evaluation_metrics.json")
print("=" * 70)
