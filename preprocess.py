import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import re

DATA_DIR = "data"
INPUT_MOVIES = f"{DATA_DIR}/movies.csv"
INPUT_USERS = f"{DATA_DIR}/users.csv"
INPUT_RATINGS = f"{DATA_DIR}/ratings.csv"

OUTPUT_MOVIES_CLEAN = f"{DATA_DIR}/movies_processed.csv"
OUTPUT_USERS_CLEAN = f"{DATA_DIR}/users_processed.csv"
OUTPUT_RATINGS_CLEAN = f"{DATA_DIR}/ratings_processed.csv"
OUTPUT_FINAL = f"{DATA_DIR}/final_dataset.csv"
OUTPUT_SEGMENTS = f"{DATA_DIR}/user_segments.csv"


def extract_year(title):
    """Trích xuất năm từ tiêu đề: 'Toy Story (1995)' -> 1995"""
    match = re.search(r'\(([-]?\d{4})\)\s*$', str(title))
    return int(match.group(1)) if match else np.nan


def clean_title(title):
    """Xóa năm khỏi tiêu đề: 'Toy Story (1995)' -> 'Toy Story'"""
    title = str(title).strip()
    title = re.sub(r'\s*\(\d{4}\)\s*$', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title.strip()


def clean_genres(genres):
    """Chuẩn hóa thể loại: xóa khoảng trắng, loại bỏ khoảng trống"""
    if pd.isna(genres):
        return ''
    genres = str(genres).strip()
    genre_list = [g.strip() for g in genres.split('|') if g.strip()]
    return '|'.join(genre_list) if genre_list else ''


def preprocess_overview(text):
    """Tiền xử lý văn bản mô tả: in thường, xóa URL, ký tự đặc biệt, khoảng trắng thừa"""
    if pd.isna(text) or not isinstance(text, str):
        return ''

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_raw_data():
    movies = pd.read_csv(INPUT_MOVIES)
    users = pd.read_csv(INPUT_USERS)
    ratings = pd.read_csv(INPUT_RATINGS)
    return movies, users, ratings


def clean_movies_data(movies_df):
    movies = movies_df.copy()

    movies['movie_id'] = pd.to_numeric(movies['movie_id'], errors='coerce')
    movies = movies.dropna(subset=['movie_id'])
    movies['movie_id'] = movies['movie_id'].astype(int)

    movies['title'] = movies['title'].fillna('').astype(str).str.strip()
    movies['title'] = movies['title'].str.replace(r'\s+', ' ', regex=True)
    movies = movies[movies['title'] != '']

    movies['year'] = movies['title'].apply(extract_year)
    movies = movies[(movies['year'] >= 1900) & (movies['year'] <= 2026)]

    movies['clean_title'] = movies['title'].apply(clean_title)
    movies['genres'] = movies['genres'].apply(clean_genres)
    movies['overview_clean'] = movies['overview'].apply(preprocess_overview)

    movies = movies[[
        'movie_id', 'title', 'clean_title', 'genres', 'year',
        'poster_url', 'overview', 'overview_clean', 'tmdb_rating', 'num_ratings', 'avg_rating'
    ]].copy()

    movies = movies.drop_duplicates(subset=['movie_id'], keep='first')
    return movies


def clean_users_data(users_df):
    users = users_df.copy()

    users['user_id'] = pd.to_numeric(users['user_id'], errors='coerce')
    users = users.dropna(subset=['user_id'])
    users['user_id'] = users['user_id'].astype(int)

    users['account'] = users['account'].fillna('').astype(str).str.strip()
    users['password'] = users['password'].fillna('').astype(str).str.strip()

    users['birth_year'] = pd.to_numeric(users['birth_year'], errors='coerce')
    users = users[(users['birth_year'] >= 1900) & (users['birth_year'] <= 2020)]
    users['birth_year'] = users['birth_year'].astype(int)

    users['gender'] = users['gender'].fillna('Unknown').astype(str).str.strip()
    valid_genders = {'Male', 'Female', 'Other', 'Unknown'}
    users['gender'] = users['gender'].apply(
        lambda x: x if x in valid_genders else 'Unknown'
    )

    users['favorite_genres'] = users['favorite_genres'].fillna('').astype(str).str.strip()
    users['favorite_genres'] = users['favorite_genres'].apply(clean_genres)
    users['favorite_genres'] = users['favorite_genres'].replace('', np.nan)

    users = users.drop_duplicates(subset=['user_id'], keep='first')
    return users


def clean_ratings_data(ratings_df):
    ratings = ratings_df.copy()

    ratings['user_id'] = pd.to_numeric(ratings['user_id'], errors='coerce')
    ratings = ratings.dropna(subset=['user_id'])
    ratings['user_id'] = ratings['user_id'].astype(int)

    ratings['movie_id'] = pd.to_numeric(ratings['movie_id'], errors='coerce')
    ratings = ratings.dropna(subset=['movie_id'])
    ratings['movie_id'] = ratings['movie_id'].astype(int)

    ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')
    ratings = ratings[(ratings['rating'] >= 1) & (ratings['rating'] <= 5)]
    ratings['rating'] = ratings['rating'].astype(float)

    ratings = ratings.drop_duplicates(subset=['user_id', 'movie_id'], keep='first')
    return ratings


def filter_consistency(movies, users, ratings):
    valid_users = set(users['user_id'].unique())
    ratings = ratings[ratings['user_id'].isin(valid_users)]

    valid_movies = set(movies['movie_id'].unique())
    ratings = ratings[ratings['movie_id'].isin(valid_movies)]

    users_with_ratings = set(ratings['user_id'].unique())
    users = users[users['user_id'].isin(users_with_ratings)]

    movies_with_ratings = set(ratings['movie_id'].unique())
    movies = movies[movies['movie_id'].isin(movies_with_ratings)]

    return movies, users, ratings


def generate_outputs(movies, users, ratings):
    movies.to_csv(OUTPUT_MOVIES_CLEAN, index=False, encoding='utf-8-sig')
    users.to_csv(OUTPUT_USERS_CLEAN, index=False, encoding='utf-8-sig')
    ratings.to_csv(OUTPUT_RATINGS_CLEAN, index=False, encoding='utf-8-sig')

    final_dataset = ratings.merge(
        movies[['movie_id', 'clean_title', 'genres', 'year', 'poster_url', 'overview_clean']],
        on='movie_id',
        how='inner'
    )
    final_dataset = final_dataset[[
        'user_id', 'movie_id', 'rating', 'clean_title', 'genres', 'year', 'poster_url', 'overview_clean'
    ]].copy()
    final_dataset.to_csv(OUTPUT_FINAL, index=False, encoding='utf-8-sig')
    return final_dataset


def create_user_segments(ratings):
    """Phân khúc theo số tương tác: Tier1_New / Tier2_Medium / Tier3_Old (theo phân vị 33% và 67%)."""
    user_interactions = ratings.groupby('user_id').size().reset_index(name='num_interactions')

    p33 = user_interactions['num_interactions'].quantile(0.33)
    p67 = user_interactions['num_interactions'].quantile(0.67)

    def assign_tier(interactions):
        if interactions <= p33:
            return 'Tier1_New'
        if interactions <= p67:
            return 'Tier2_Medium'
        return 'Tier3_Old'

    user_interactions['tier'] = user_interactions['num_interactions'].apply(assign_tier)
    user_interactions.to_csv(OUTPUT_SEGMENTS, index=False, encoding='utf-8-sig')
    return user_interactions


def print_summary(movies, users, ratings, final_dataset):
    print(
        f"Hoàn tất tiền xử lý — phim: {len(movies)}, người dùng: {len(users)}, "
        f"đánh giá: {len(ratings)}, final_dataset: {len(final_dataset)} dòng."
    )


def main():
    try:
        movies, users, ratings = load_raw_data()
        movies = clean_movies_data(movies)
        users = clean_users_data(users)
        ratings = clean_ratings_data(ratings)
        movies, users, ratings = filter_consistency(movies, users, ratings)

        final_dataset = generate_outputs(movies, users, ratings)
        create_user_segments(ratings)
        print_summary(movies, users, ratings, final_dataset)
    except Exception as e:
        print(f"LỖI: {e}")


if __name__ == '__main__':
    main()
