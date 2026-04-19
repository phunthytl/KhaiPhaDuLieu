import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("TRAINING RECOMMENDATION MODELS")

os.makedirs("model", exist_ok=True)

# LOAD DATA
print("\n1. Loading data...")
final_dataset = pd.read_csv("data/final_dataset.csv")
movies_df = pd.read_csv("data/movies_processed.csv")
users_df = pd.read_csv("data/users_processed.csv")

from datetime import datetime
current_year = datetime.now().year
users_df["age"] = current_year - users_df["birth_year"]


# =================================================================
# 1. COLLABORATIVE FILTERING (MATRIX FACTORIZATION - SVD)
# =================================================================
print("\n2. Training CF Model (Matrix Factorization - SVD)...")

np.random.seed(42)
test_mask = np.random.rand(len(final_dataset)) < 0.2
train_data = final_dataset[~test_mask]
test_data = final_dataset[test_mask]

# Tạo ma trận User-Item từ tập Train
train_pivot = train_data.pivot_table(
    index='user_id', columns='movie_id', values='rating', fill_value=0
)

R = train_pivot.values
user_means = np.mean(R, axis=1)
# Trừ mean để tránh thiên lệch
R_demeaned = R - user_means.reshape(-1, 1)

# Áp dụng Truncated SVD (K=50)
k_components = min(50, R.shape[0]-1, R.shape[1]-1)
svd = TruncatedSVD(n_components=k_components, random_state=42)
svd.fit(R_demeaned)

U_Sigma = svd.transform(R_demeaned)
Vt = svd.components_

# Tái tạo ma trận dự đoán hoàn chỉnh
R_predicted = np.dot(U_Sigma, Vt) + user_means.reshape(-1, 1)
R_predicted = np.clip(R_predicted, 1, 5)

# Predict test set
y_true, y_pred = [], []
cf_predictions = {}

user_list = list(train_pivot.index)
movie_list = list(train_pivot.columns)

for _, row in test_data.iterrows():
    user_id, movie_id, true_rating = row['user_id'], row['movie_id'], row['rating']
    
    if user_id not in user_list or movie_id not in movie_list:
        pred = 3.0  # Cold start
    else:
        user_idx = user_list.index(user_id)
        movie_idx = movie_list.index(movie_id)
        # Lấy điểm trực tiếp từ ma trận SVD
        pred = R_predicted[user_idx, movie_idx]
        
    pred = np.clip(pred, 1, 5)
    
    if user_id not in cf_predictions:
        cf_predictions[user_id] = {}
    cf_predictions[user_id][movie_id] = pred
    y_true.append(true_rating)
    y_pred.append(pred)

cf_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
cf_mae = mean_absolute_error(y_true, y_pred)

# Lưu model CF với ma trận đã khử nhiễu bằng SVD
cf_model = {
    "user_list": user_list,
    "movie_list": movie_list,
    "train_matrix": R_predicted,
    "rmse": cf_rmse,
    "mae": cf_mae
}
with open("model/cf_model.pkl", "wb") as f:
    pickle.dump(cf_model, f)
print(f"CF Model (SVD): RMSE={cf_rmse:.4f}, MAE={cf_mae:.4f}")


# =================================================================
# 2. CONTENT-BASED
# =================================================================
print("\n3. Training Content-Based Model...")

movies = movies_df[["movie_id", "genres", "overview_clean", "year"]].copy()
movies = movies.dropna(subset=["genres"])
movies["overview_clean"] = movies["overview_clean"].fillna("")

vec_genres = TfidfVectorizer(analyzer="char", ngram_range=(2,2), max_features=100)
tfidf_genres = vec_genres.fit_transform(movies["genres"].astype(str))

vec_overview = TfidfVectorizer(max_features=50, min_df=2, max_df=0.8, stop_words='english')
tfidf_overview = vec_overview.fit_transform(movies["overview_clean"].astype(str))

year_norm = (movies["year"] - movies["year"].min()) / (movies["year"].max() - movies["year"].min() + 1)

features = np.hstack([
    tfidf_genres.toarray() * 0.5,
    tfidf_overview.toarray() * 0.4,
    year_norm.values.reshape(-1, 1) * 0.1
])

features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)

user_profiles = {}
for user_id in final_dataset["user_id"].unique():
    user_ratings = final_dataset[final_dataset["user_id"] == user_id]
    rated_idx = []
    
    for _, row in user_ratings.iterrows():
        movie_id = row["movie_id"]
        movie_idx = movies[movies["movie_id"] == movie_id].index.tolist()
        if movie_idx:
            rated_idx.append(movie_idx[0])
    
    if rated_idx:
        user_profiles[user_id] = features_norm[rated_idx].mean(axis=0)
    else:
        user_profiles[user_id] = np.zeros(features_norm.shape[1])

content_scores = {}
for user_id, profile in user_profiles.items():
    sims = cosine_similarity(profile.reshape(1, -1), features_norm).flatten()
    content_scores[user_id] = {
        movie_id: (sims[idx] + 1) / 2.0
        for idx, movie_id in enumerate(movies["movie_id"].values)
    }

content_model = {
    "vec_genres": vec_genres,
    "vec_overview": vec_overview,
    "features_norm": features_norm,
    "movie_ids": movies["movie_id"].values,
    "user_profiles": user_profiles,
    "content_scores": content_scores
}
with open("model/content_model.pkl", "wb") as f:
    pickle.dump(content_model, f)
print(f"Content-Based Model: {features_norm.shape[1]} features")


# =================================================================
# 3. DEMOGRAPHIC-BASED
# =================================================================
print("\n4. Training Demographic Model...")

merged = final_dataset.merge(users_df[["user_id", "gender", "age"]], on="user_id")
merged = merged.merge(movies_df[["movie_id", "genres", "year"]], on="movie_id", how="left")

merged["age_group"] = pd.cut(
    merged["age"], bins=[0, 18, 25, 35, 50, 100],
    labels=["<18", "18-25", "25-35", "35-50", "50+"]
)

demo_patterns = {}
for gender in merged["gender"].unique():
    demo_patterns[gender] = {}
    for age_group in merged["age_group"].unique():
        subset = merged[(merged["gender"] == gender) & (merged["age_group"] == age_group)]
        if len(subset) > 0:
            demo_patterns[gender][age_group] = {"avg_rating": subset["rating"].mean()}

demo_scores = {}
for user_id in final_dataset["user_id"].unique():
    user_info = users_df[users_df["user_id"] == user_id].iloc[0]
    gender, age = user_info["gender"], user_info["age"]
    
    age_group = ("<18" if age < 18 else "18-25" if age < 25 else 
                 "25-35" if age < 35 else "35-50" if age < 50 else "50+")
    
    baseline = demo_patterns.get(gender, {}).get(age_group, {}).get("avg_rating", 3.0)
    
    demo_scores[user_id] = {
        movie_id: np.clip(baseline/5.0 + np.random.normal(0, 0.05), 0, 1)
        for movie_id in final_dataset["movie_id"].unique()
    }

demo_model = {
    "patterns": demo_patterns,
    "scores": demo_scores,
    "age_groups": ["<18", "18-25", "25-35", "35-50", "50+"]
}
with open("model/demographic_model.pkl", "wb") as f:
    pickle.dump(demo_model, f)
print(f"Demographic Model: {len(demo_patterns)} gender groups")

# SUMMARY
print("\nTRAINING COMPLETE")
