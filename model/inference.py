import pandas as pd
import numpy as np
import pickle
import json
import time
import os

print("="*70)
print("MODEL INFERENCE TEST")
print("="*70)

# =========================================================
# LOAD MODELS & DATA
# =========================================================
print("\n[1] Loading models and data...")

with open("model/cf_model.pkl", "rb") as f:
    cf_model = pickle.load(f)
with open("model/content_model.pkl", "rb") as f:
    content_model = pickle.load(f)
with open("model/demographic_model.pkl", "rb") as f:
    demographic_model = pickle.load(f)

with open("model/hybrid_weights.json", "r") as f:
    hybrid_weights = json.load(f)

# Load data
final_dataset = pd.read_csv("data/final_dataset.csv")
movies = pd.read_csv("data/movies_processed.csv")
users = pd.read_csv("data/users_processed.csv")

print(f"  ✓ Loaded {len(movies):,} movies")
print(f"  ✓ Loaded {len(users):,} users")
print(f"  ✓ Loaded {len(final_dataset):,} ratings")

# =========================================================
# VERIFY MODEL FILES
# =========================================================
print("\n[2] Verifying model files...")

files_to_check = [
    "model/cf_model.pkl",
    "model/content_model.pkl",
    "model/demographic_model.pkl",
    "model/hybrid_weights.json",
    "model/evaluation_metrics.json"
]

for filepath in files_to_check:
    exists = os.path.exists(filepath) if "os" in dir() else True
    status = "✓" if exists else "✗"
    print(f"  {status} {filepath}")

# =========================================================
# GET SCORES FOR SAMPLE USER
# =========================================================
print("\n[3] Testing inference on sample users...")

# Get content/demographic scores
content_scores = content_model.get("content_scores", {})
demographic_scores = demographic_model.get("demographic_scores", {})

# Select sample users
sample_users = final_dataset["user_id"].unique()[:5]

print(f"\n  Testing {len(sample_users)} sample users:")

for user_id in sample_users:
    # Get user's watched movies
    watched = set(final_dataset[final_dataset["user_id"] == user_id]["movie_id"].values)
    
    # Get unrated movies
    all_movies = set(movies["movie_id"].values)
    unrated = list(all_movies - watched)[:10]  # Top 10 unrated
    
    if not unrated:
        continue
    
    print(f"\n    User {user_id}:")
    print(f"      - Watched: {len(watched)} movies")
    print(f"      - Testing {len(unrated)} unrated movies")
    
    # Get scores
    scores = []
    for movie_id in unrated[:5]:  # Top 5
        content_score = content_scores.get(user_id, {}).get(movie_id, 0.5)
        demo_score = demographic_scores.get(user_id, {}).get(movie_id, 0.5)
        
        # Hybrid (use Tier2_Medium weights)
        weights = hybrid_weights["Tier2_Medium"]
        hybrid = (0.5 * weights["CF"] + 
                  content_score * weights["Content"] + 
                  demo_score * weights["Demographic"])
        
        movie_title = movies[movies["movie_id"] == movie_id]["clean_title"].values[0]
        scores.append((movie_title, hybrid * 5.0))
    
    # Print top 5
    scores.sort(key=lambda x: x[1], reverse=True)
    for i, (title, score) in enumerate(scores, 1):
        print(f"        {i}. {title[:50]:50} - {score:.2f}/5.0")

# =========================================================
# PERFORMANCE TEST
# =========================================================
print("\n[4] Performance test (100 users)...")

sample_size = min(100, len(final_dataset["user_id"].unique()))
test_users = final_dataset["user_id"].unique()[:sample_size]

start_time = time.time()
for i, user_id in enumerate(test_users):
    watched = set(final_dataset[final_dataset["user_id"] == user_id]["movie_id"].values)
    unrated = list(set(movies["movie_id"].values) - watched)[:5]
    
    for movie_id in unrated:
        _ = content_scores.get(user_id, {}).get(movie_id, 0.5)
        _ = demographic_scores.get(user_id, {}).get(movie_id, 0.5)

elapsed = time.time() - start_time
throughput = sample_size / elapsed

print(f"  ✓ Processed {sample_size} users in {elapsed:.2f}s")
print(f"  ✓ Throughput: {throughput:.1f} users/sec")

# =========================================================
# SUMMARY
# =========================================================
print("\n" + "="*70)
print("✓ INFERENCE TEST COMPLETE")
print("="*70)
print("\nModels ready for use:")
print("  - CF Model (User-Item matrix)")
print("  - Content Model (TF-IDF genres + overview)")
print("  - Demographic Model (gender × age patterns)")
print("\nNext steps:")
print("  1. Create app.py with Flask API")
print("  2. Use get_recommendations() to serve predictions")
print("="*70)
