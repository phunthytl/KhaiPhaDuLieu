"""
Chạy từ thư mục gốc project hoặc bất kỳ đâu: python test_hybrid.py [user_id]

- Không có tham số: nhập user_id trên terminal.
- Có tham số: dùng user_id đó (ví dụ: python test_hybrid.py 42).

Hiển thị: phim đã đánh giá (rating) và top phim gợi ý (chưa xem).
"""
import io
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

# UTF-8 trên Windows
_enc = getattr(sys.stdout, "encoding", None) or ""
if hasattr(sys.stdout, "buffer") and _enc.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
MODEL = os.path.join(ROOT, "model")


def read_user_id() -> int:
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1].strip())
        except ValueError:
            print("Tham số user_id phải là số nguyên.", file=sys.stderr)
            sys.exit(1)
    try:
        s = input("Nhập user_id: ").strip()
        return int(s)
    except EOFError:
        print("Không có dữ liệu nhập.", file=sys.stderr)
        sys.exit(1)
    except ValueError:
        print("user_id phải là số nguyên.", file=sys.stderr)
        sys.exit(1)


def load_assets():
    movies = pd.read_csv(os.path.join(DATA, "movies_processed.csv"))
    user_segments = pd.read_csv(os.path.join(DATA, "user_segments.csv"))
    ratings = pd.read_csv(os.path.join(DATA, "ratings_processed.csv"))

    with open(os.path.join(MODEL, "cf_model.pkl"), "rb") as f:
        cf_model = pickle.load(f)
    with open(os.path.join(MODEL, "content_model.pkl"), "rb") as f:
        content_model = pickle.load(f)
    with open(os.path.join(MODEL, "demographic_model.pkl"), "rb") as f:
        demographic_model = pickle.load(f)
    with open(os.path.join(MODEL, "hybrid_weights.json"), "r", encoding="utf-8") as f:
        hybrid_weights = json.load(f)

    return movies, user_segments, ratings, cf_model, content_model, demographic_model, hybrid_weights


def print_rated_movies(user_id: int, ratings: pd.DataFrame, movies: pd.DataFrame):
    ur = ratings[ratings["user_id"] == user_id][["movie_id", "rating"]].copy()
    if ur.empty:
        print("\n--- Phim đã đánh giá ---")
        print("(Không có dữ liệu rating cho user này.)")
        return

    ur = ur.merge(
        movies[["movie_id", "clean_title", "title", "genres"]],
        on="movie_id",
        how="left",
    )
    ur["hiển_thị"] = ur["clean_title"].fillna(ur["title"]).fillna("(không tên)")
    ur = ur.sort_values(["rating", "hiển_thị"], ascending=[False, True])

    print("\n--- Phim đã đánh giá ---")
    print(f"Tổng: {len(ur)} phim\n")
    for _, row in ur.iterrows():
        title = row["hiển_thị"]
        g = row["genres"] if pd.notna(row["genres"]) else ""
        print(f"  [{row['movie_id']:5d}] {title}  |  {row['rating']:.1f}★  |  {g}")


def build_recommendations(
    user_id: int,
    tier: str,
    weights: dict,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    cf_model: dict,
    content_scores: dict,
    demo_scores: dict,
    top_k: int,
):
    cf_user_list = cf_model.get("user_list", [])
    cf_movie_list = cf_model.get("movie_list", [])
    cf_matrix = cf_model.get("train_matrix", np.array([]))

    def get_cf_score(u_id, m_id):
        if u_id not in cf_user_list or m_id not in cf_movie_list:
            return 0.5
        u_idx = cf_user_list.index(u_id)
        m_idx = cf_movie_list.index(m_id)
        try:
            s = cf_matrix[u_idx, m_idx]
            return float(np.clip(s / 5.0 if s > 0 else 0.5, 0, 1))
        except Exception:
            return 0.5

    watched = set(ratings[ratings["user_id"] == user_id]["movie_id"].values)
    all_movies = movies["movie_id"].unique()
    unwatched = [m for m in all_movies if m not in watched]

    w_cf = float(weights["CF"])
    w_ct = float(weights["Content"])
    w_dm = float(weights["Demographic"])

    rows = []
    for m_id in unwatched:
        cf_s = get_cf_score(user_id, m_id)
        cont_s = float(content_scores.get(user_id, {}).get(m_id, 0.5))
        demo_s = float(demo_scores.get(user_id, {}).get(m_id, 0.5))
        hybrid_s = cf_s * w_cf + cont_s * w_ct + demo_s * w_dm
        pred = float(np.clip(hybrid_s, 0, 1) * 5.0)
        rows.append((m_id, pred, cf_s, cont_s, demo_s))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]


def print_recommendations(user_id: int, top_rows, movies: pd.DataFrame):
    print("\n--- Top gợi ý (phim chưa xem) ---\n")
    if not top_rows:
        print("(Không còn phim chưa xem.)")
        return

    movie_lookup = movies.set_index("movie_id")
    for i, (m_id, pred, cf_s, ct_s, dm_s) in enumerate(top_rows, 1):
        if m_id not in movie_lookup.index:
            title = f"(id {m_id})"
            genres = ""
        else:
            r = movie_lookup.loc[m_id]
            title = r.get("clean_title") or r.get("title") or str(m_id)
            genres = r.get("genres", "") if pd.notna(r.get("genres")) else ""
        print(f"  {i:2d}. [{m_id:5d}] {title}")
        print(f"      Dự đoán: {pred:.2f}/5  (CF={cf_s:.2f}, Content={ct_s:.2f}, Demo={dm_s:.2f})  |  {genres}")


def main():
    user_id = read_user_id()

    try:
        movies, user_segments, ratings, cf_model, content_model, demographic_model, hybrid_weights = load_assets()
    except FileNotFoundError as e:
        print(f"Thiếu file dữ liệu hoặc model: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi tải: {e}", file=sys.stderr)
        sys.exit(1)

    info = user_segments[user_segments["user_id"] == user_id]
    if info.empty:
        print(f"Không có user_id={user_id} trong user_segments.csv.")
        sys.exit(1)

    tier = str(info["tier"].iloc[0])
    n_int = int(info["num_interactions"].iloc[0]) if "num_interactions" in info.columns else 0
    weights = hybrid_weights.get(tier) or hybrid_weights.get("Tier2_Medium", {"CF": 0.34, "Content": 0.33, "Demographic": 0.33})

    print("=" * 72)
    print(f"User {user_id}  |  Tier: {tier}  |  Số tương tác (theo segment): {n_int}")
    print(f"Trọng số hybrid: CF={weights['CF']}, Content={weights['Content']}, Demographic={weights['Demographic']}")
    print("=" * 72)

    print_rated_movies(user_id, ratings, movies)

    content_scores = content_model.get("content_scores", {})
    demo_scores = demographic_model.get("demographic_scores", {})

    top_k = 15
    if len(sys.argv) > 2:
        try:
            top_k = max(1, int(sys.argv[2]))
        except ValueError:
            pass

    top_rows = build_recommendations(
        user_id,
        tier,
        weights,
        movies,
        ratings,
        cf_model,
        content_scores,
        demo_scores,
        top_k=top_k,
    )
    print_recommendations(user_id, top_rows, movies)
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
