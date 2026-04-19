import pandas as pd
import numpy as np
import pickle
import json
import sys

def recommend_for_user(user_id, top_k=10):
    print("="*70)
    print(f"BỘ KHUYẾN NGHỊ LAI (HYBRID RECOMMENDER) - USER {user_id}")
    print("="*70)
    
    # 1. Tải dữ liệu
    try:
        movies = pd.read_csv("data/movies_processed.csv")
        user_segments = pd.read_csv("data/user_segments.csv")
        ratings = pd.read_csv("data/ratings_processed.csv")
    except Exception as e:
        print(f"Lỗi tải file CSV: {e}")
        return

    # 2. Tải model
    try:
        with open("model/cf_model.pkl", "rb") as f:
            cf_model = pickle.load(f)
        with open("model/content_model.pkl", "rb") as f:
            content_model = pickle.load(f)
        with open("model/demographic_model.pkl", "rb") as f:
            demographic_model = pickle.load(f)
        with open("model/hybrid_weights.json", "r", encoding="utf-8") as f:
            hybrid_weights = json.load(f)
    except Exception as e:
        print(f"Lỗi tải file model: {e}")
        print("Vui lòng đảm bảo bạn đã chạy quá trình huấn luyện các mô hình trước.")
        return

    # 3. Lấy thông tin user
    user_info = user_segments[user_segments["user_id"] == user_id]
    if user_info.empty:
        print(f"❌ Không tìm thấy user {user_id} trong hệ thống.")
        return
    
    tier = user_info["tier"].values[0]
    num_interactions = user_info["num_interactions"].values[0]
    weights = hybrid_weights.get(tier, hybrid_weights["Tier2_Medium"])
    
    print(f"👤 User ID: {user_id}")
    print(f"📊 Số lượt đánh giá (tương tác): {num_interactions}")
    print(f"🏷️ Phân khúc (Tier): {tier}")
    print(f"⚖️ Trọng số sử dụng: CF={weights['CF']}, Content={weights['Content']}, Demographic={weights['Demographic']}")
    print("-" * 70)

    # Các bộ điểm (scores)
    content_scores = content_model.get("content_scores", {})
    demo_scores = demographic_model.get("demographic_scores", {})
    
    cf_user_list = cf_model.get("user_list", [])
    cf_movie_list = cf_model.get("movie_list", [])
    cf_matrix = cf_model.get("train_matrix", np.array([]))

    # Xây dựng hàm lấy điểm CF
    def get_cf_score(u_id, m_id):
        if u_id not in cf_user_list or m_id not in cf_movie_list:
            return 0.5
        u_idx = cf_user_list.index(u_id)
        m_idx = cf_movie_list.index(m_id)
        try:
            s = cf_matrix[u_idx, m_idx]
            return np.clip(s / 5.0 if s > 0 else 0.5, 0, 1)
        except:
            return 0.5

    # 4. Tính toán cho tất cả các phim
    watched_movies = set(ratings[ratings["user_id"] == user_id]["movie_id"].values)
    all_movies = movies["movie_id"].unique()
    
    print(f"Đang phân tích {len(all_movies) - len(watched_movies)} phim chưa xem của người dùng...")
    recommendations = []
    
    unwatched_movies = [m for m in all_movies if m not in watched_movies]
    
    for m_id in unwatched_movies:
        # 1. Collaborative Filtering Score
        cf_s = 0.5 if tier == "Tier1_New" else get_cf_score(user_id, m_id)
        
        # 2. Content-based Score
        cont_s = content_scores.get(user_id, {}).get(m_id, 0.5)
        
        # 3. Demographic Score
        demo_s = demo_scores.get(user_id, {}).get(m_id, 0.5)
        
        # Hybrid Score (Tổng hợp)
        hybrid_s = (cf_s * weights["CF"] + 
                    cont_s * weights["Content"] + 
                    demo_s * weights["Demographic"])
        
        # Quy đổi từ thang 0-1 sang thang 0-5
        predicted_rating = np.clip(hybrid_s, 0, 1) * 5.0
        
        recommendations.append({
            "movie_id": m_id,
            "cf_score": cf_s,
            "content_score": cont_s,
            "demo_score": demo_s,
            "predicted_rating": predicted_rating
        })
        
    # Sắp xếp top K
    recommendations.sort(key=lambda x: x["predicted_rating"], reverse=True)
    top_recs = recommendations[:top_k]
    
    print("\n🎬 TOP KẾT QUẢ GỢI Ý:")
    print("="*70)
    for i, rec in enumerate(top_recs, 1):
        m_id = rec["movie_id"]
        movie_row = movies[movies["movie_id"] == m_id]
        if movie_row.empty:
            continue
            
        title = movie_row["title"].values[0]
        genres = movie_row["genres"].values[0]
        score = rec["predicted_rating"]
        
        print(f" {i:2d}. {title}")
        print(f"     ► Thể loại: {genres}")
        print(f"     ► Điểm dự đoán: {score:.2f} ⭐ (Chi tiết: CF={rec['cf_score']:.2f}, Cont={rec['content_score']:.2f}, Demo={rec['demo_score']:.2f})")
        print()

if __name__ == "__main__":
    # Test mặc định với user ID = 1, bạn có thể truyền ID khác qua command line
    # Ví dụ: python test_hybrid.py 5
    user_to_test = 1
    if len(sys.argv) > 1:
        try:
            user_to_test = int(sys.argv[1])
        except ValueError:
            print("Vui lòng nhập User ID là một số nguyên.")
            sys.exit(1)
            
    recommend_for_user(user_to_test, top_k=10)
