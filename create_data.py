import pandas as pd
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import hashlib
import random
from collections import Counter
from datetime import datetime

# =========================================================
# CẤU HÌNH
# =========================================================
DATA_DIR = "data"
RATINGS_FILE = f"{DATA_DIR}/ratings.csv"
MOVIES_FILE = f"{DATA_DIR}/movies.csv"
OUTPUT_USERS = f"{DATA_DIR}/users.csv"
OUTPUT_RATINGS = f"{DATA_DIR}/ratings_clean.csv"

BIRTH_YEAR_MIN = 1945
BIRTH_YEAR_MAX = 2010

GENDERS = ["Male", "Female", "Other"]

# =========================================================
# HỖ TRỢ HÀM
# =========================================================
def generate_password(username, length=12):
    """Sinh password từ username (deterministic)"""
    # Hash username để tạo password consistent
    hashed = hashlib.md5((username + "seed123").encode()).hexdigest()
    # Tạo password mix uppercase, lowercase, numbers, special chars
    chars = hashed[:length-3]
    password = chars.upper()[:3] + chars.lower()[:3] + chars[6:length-6]
    return password[:length]

def get_favorite_genres(user_ratings, movies_df, top_n=3):
    """
    Lấy thể loại yêu thích của user dựa trên ratings cao nhất
    
    Args:
        user_ratings: DataFrame với columns [movie_id, rating]
        movies_df: DataFrame movies với columns [movie_id, genres]
        top_n: Số phim cao nhất để xem
    
    Returns:
        Chuỗi genres cách nhau bằng |, hoặc None nếu không có rating cao
    """
    
    # Lấy top N phim có rating cao nhất
    top_movies = user_ratings.nlargest(top_n, 'rating')
    
    if len(top_movies) == 0:
        return None
    
    # Lấy genres từ những phim này
    favorite_genres = []
    
    for idx, row in top_movies.iterrows():
        movie_id = row['movie_id']
        movie = movies_df[movies_df['movie_id'] == movie_id]
        
        if not movie.empty:
            genres = movie.iloc[0]['genres']
            if pd.notna(genres):
                # Split genres và thêm vào list
                for genre in str(genres).split('|'):
                    favorite_genres.append(genre.strip())
    
    # Đếm genres xuất hiện
    if favorite_genres:
        genre_counts = Counter(favorite_genres)
        # Lấy top 3 genres theo frequency
        top_genres = [g for g, _ in genre_counts.most_common(3)]
        return '|'.join(top_genres) if top_genres else None
    
    return None

# =========================================================
# MAIN PROCESS
# =========================================================
def main():
    print("=" * 70)
    print("CREATE USER DATA AND SIMPLIFY RATINGS")
    print("=" * 70)
    
    # ===== STEP 1: Load data =====
    print("\n[Step 1] Loading data...")
    
    print("  Loading movies.csv...")
    movies_df = pd.read_csv(MOVIES_FILE)
    print(f"  OK: Loaded {len(movies_df)} movies")
    
    print("  Loading ratings.csv (large file)...")
    ratings_df = pd.read_csv(RATINGS_FILE)
    print(f"  OK: Loaded {len(ratings_df)} ratings")
    
    # ===== STEP 2: Create user data =====
    print("\n[Step 2] Creating user data...")
    
    # Get unique users from ratings
    unique_users = sorted(ratings_df['user_id'].unique())
    num_users = len(unique_users)
    print(f"  Found {num_users} users in ratings file")
    
    users_data = []
    
    for i, user_id in enumerate(unique_users):
        # Progress
        if (i + 1) % 1000 == 0:
            print(f"  Processing {i + 1}/{num_users} users...")
        
        # Create account (user + ID)
        account = f"user_{user_id:06d}"
        
        # Create password
        password = generate_password(account)
        
        # Create birth year (random, consistent based on user_id)
        random.seed(int(user_id))  # Deterministic random
        birth_year = random.randint(BIRTH_YEAR_MIN, BIRTH_YEAR_MAX)
        
        # Create gender
        gender = random.choice(GENDERS)
        
        # Calculate favorite genres from user ratings
        user_ratings = ratings_df[ratings_df['user_id'] == user_id][['movie_id', 'rating']]
        favorite_genres = get_favorite_genres(user_ratings, movies_df, top_n=5)
        
        # Make realistic: 70% users have favorite genres, 30% null
        if random.random() < 0.3:
            favorite_genres = None
        
        users_data.append({
            'user_id': user_id,
            'account': account,
            'password': password,
            'birth_year': birth_year,
            'gender': gender,
            'favorite_genres': favorite_genres
        })
    
    users_df = pd.DataFrame(users_data)
    print(f"  OK: Created {len(users_df)} user records")
    
    # ===== STEP 3: Save users.csv =====
    print(f"\n[Step 3] Saving users.csv ({OUTPUT_USERS})...")
    
    users_df.to_csv(OUTPUT_USERS, index=False)
    print(f"  OK: Saved {len(users_df)} users")
    
    # Statistics
    print("\n  [USER STATS]")
    print(f"    Total users: {len(users_df)}")
    
    # Gender distribution
    gender_dist = users_df['gender'].value_counts()
    print(f"    Gender:")
    for gender, count in gender_dist.items():
        pct = (count / len(users_df)) * 100
        print(f"      {gender}: {count} ({pct:.1f}%)")
    
    # Birth year distribution
    print(f"    Birth year:")
    print(f"      Min: {users_df['birth_year'].min()}")
    print(f"      Max: {users_df['birth_year'].max()}")
    print(f"      Mean: {users_df['birth_year'].mean():.1f}")
    
    # Favorite genres distribution
    with_genres = users_df['favorite_genres'].notna().sum()
    without_genres = users_df['favorite_genres'].isna().sum()
    print(f"    Favorite genres:")
    print(f"      With genres: {with_genres} ({(with_genres/len(users_df))*100:.1f}%)")
    print(f"      Null genres: {without_genres} ({(without_genres/len(users_df))*100:.1f}%)")
    
    # ===== STEP 4: Simplify ratings.csv =====
    print(f"\n[Step 4] Simplifying ratings.csv...")
    
    # Keep only 3 columns: user_id, movie_id, rating
    ratings_clean = ratings_df[['user_id', 'movie_id', 'rating']].copy()
    
    # Save
    ratings_clean.to_csv(OUTPUT_RATINGS, index=False)
    print(f"  OK: Saved {len(ratings_clean)} ratings (columns: user_id, movie_id, rating)")
    
    # Statistics
    print("\n  [RATINGS STATS]")
    print(f"    Total ratings: {len(ratings_clean)}")
    print(f"    Total users: {ratings_clean['user_id'].nunique()}")
    print(f"    Total movies: {ratings_clean['movie_id'].nunique()}")
    print(f"    Average rating: {ratings_clean['rating'].mean():.2f}")
    
    rating_dist = ratings_clean['rating'].value_counts().sort_index()
    print(f"    Rating distribution:")
    for rating, count in rating_dist.items():
        pct = (count / len(ratings_clean)) * 100
        print(f"      {rating}: {count:,} ({pct:.1f}%)")
    
    # ===== DONE =====
    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  1. {OUTPUT_USERS}")
    print(f"     {len(users_df)} users")
    print(f"     Columns: user_id, account, password, birth_year, gender, favorite_genres")
    print(f"\n  2. {OUTPUT_RATINGS}")
    print(f"     {len(ratings_clean)} ratings")
    print(f"     Columns: user_id, movie_id, rating")
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
