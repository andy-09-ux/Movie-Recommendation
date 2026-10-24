
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------
# OMDb SETTINGS
# ---------------------------------------------------------
# Option A: Set your key in the environment variable OMDB_API_KEY.
# Option B: Replace the placeholder below with your own key.
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")

OMDB_URL = "https://www.omdbapi.com/"
poster_cache = {}


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------
csv_path = BASE_DIR / "movies.csv"

if not csv_path.exists():
    raise FileNotFoundError(
        "movies.csv must be in the same folder as app.py."
    )

movies = pd.read_csv(csv_path)

required_columns = {"title", "genres"}
if not required_columns.issubset(movies.columns):
    raise ValueError(
        "movies.csv must contain 'title' and 'genres' columns."
    )

movies = movies.dropna(subset=["title", "genres"]).copy()
movies["title"] = movies["title"].astype(str).str.strip()
movies["genres"] = movies["genres"].astype(str)
movies = movies[
    (movies["title"] != "") &
    (movies["title"].str.lower() != "nan")
]
movies = movies.drop_duplicates(subset=["title"]).reset_index(drop=True)

movies["genres_clean"] = (
    movies["genres"]
    .replace("(no genres listed)", "")
    .str.replace("|", " ", regex=False)
)


# ---------------------------------------------------------
# MOOD-TO-GENRE MAPPING
# ---------------------------------------------------------
MOOD_GENRES = {
    "Happy": ["Comedy", "Animation", "Children", "Musical"],
    "Romantic": ["Romance", "Comedy"],
    "Excited": ["Action", "Adventure", "Thriller", "Sci-Fi"],
    "Suspenseful": ["Thriller", "Mystery", "Horror", "Crime"],
    "Scared": ["Horror", "Thriller"],
    "Emotional": ["Drama", "Romance"],
    "Relaxed": ["Animation", "Comedy", "Children", "Musical"],
    "Curious": ["Documentary", "Mystery", "Sci-Fi"],
    "Serious": ["Drama", "Documentary", "Crime"],
    "Adventurous": ["Adventure", "Action", "Fantasy"]
}


def get_movie_moods(genre_string):
    movie_genres = genre_string.split("|")
    matched_moods = []

    for mood, related_genres in MOOD_GENRES.items():
        if any(genre in movie_genres for genre in related_genres):
            matched_moods.append(mood)

    return matched_moods


movies["moods"] = movies["genres"].apply(get_movie_moods)


# ---------------------------------------------------------
# BUILD TF-IDF MODEL ONCE
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    movies["genres_clean"].fillna("")
)

all_genres = sorted({
    genre
    for genre_string in movies["genres"]
    for genre in genre_string.split("|")
    if genre and genre != "(no genres listed)"
})

all_moods = sorted(MOOD_GENRES.keys())


# ---------------------------------------------------------
# OMDb POSTER LOOKUP
# ---------------------------------------------------------
def get_movie_poster(title):
    if title in poster_cache:
        return poster_cache[title]

    if (
        not OMDB_API_KEY or
        OMDB_API_KEY == "PASTE_YOUR_OMDB_API_KEY_HERE"
    ):
        return ""

    # MovieLens titles often look like "Toy Story (1995)".
    year_match = re.search(r"\((\d{4})\)$", title)
    year = year_match.group(1) if year_match else None
    clean_title = re.sub(r"\s*\(\d{4}\)$", "", title).strip()

    params = {
        "apikey": OMDB_API_KEY,
        "t": clean_title,
        "type": "movie"
    }
    if year:
        params["y"] = year

    try:
        response = requests.get(
            OMDB_URL,
            params=params,
            timeout=8
        )
        response.raise_for_status()
        data = response.json()

        poster = data.get("Poster", "")
        if (
            data.get("Response") == "True" and
            poster and
            poster != "N/A"
        ):
            poster_cache[title] = poster
            return poster

    except (requests.RequestException, ValueError) as error:
        app.logger.warning(
            "Poster lookup failed for %s: %s", title, error
        )

    poster_cache[title] = ""
    return ""


# ---------------------------------------------------------
# RECOMMENDATION ENGINE
# ---------------------------------------------------------
def recommend_movies(selected_genre, selected_mood, limit=6):
    # Create the search vector once per request.
    mood_genres = MOOD_GENRES[selected_mood]
    query_text = " ".join([selected_genre] + mood_genres)
    query_vector = vectorizer.transform([query_text])

    similarity_scores = cosine_similarity(
        query_vector, tfidf_matrix
    ).ravel()

    recommendations = []

    for index, movie in movies.iterrows():
        movie_genres = movie["genres"].split("|")
        movie_moods = movie["moods"]

        # Exact genre match
        genre_score = float(selected_genre in movie_genres)

        # Mood match, with a partial score for related genres
        if selected_mood in movie_moods:
            mood_score = 1.0
        else:
            related = set(mood_genres)
            matching = len(related.intersection(movie_genres))
            mood_score = matching / len(related) if related else 0.0

        similarity_score = float(similarity_scores[index])

        # Weighted recommendation score
        final_score = (
            0.50 * genre_score +
            0.30 * mood_score +
            0.20 * similarity_score
        )

        recommendations.append({
            "title": movie["title"],
            "genres": movie["genres"],
            "moods": movie_moods,
            "raw_score": final_score
        })

    recommendations.sort(
        key=lambda item: item["raw_score"],
        reverse=True
    )

    # Keep the top unique titles
    top_movies = []
    seen_titles = set()

    for movie in recommendations:
        if movie["title"] in seen_titles:
            continue

        seen_titles.add(movie["title"])
        top_movies.append(movie)

        if len(top_movies) >= limit:
            break

    # Fetch posters only for movies being displayed
    for movie in top_movies:
        movie["poster"] = get_movie_poster(movie["title"])
        movie["score"] = round(movie.pop("raw_score") * 100, 2)

    return top_movies


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        genres=all_genres,
        moods=all_moods
    )


# ---------------------------------------------------------
# RECOMMENDATION API
# ---------------------------------------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(silent=True) or {}

    selected_genre = data.get("genre", "")
    selected_mood = data.get("mood", "")

    if selected_genre not in all_genres:
        return jsonify({"error": "Please select a valid genre."}), 400

    if selected_mood not in all_moods:
        return jsonify({"error": "Please select a valid mood."}), 400

    results = recommend_movies(
        selected_genre,
        selected_mood,
        limit=6
    )

    return jsonify({
        "genre": selected_genre,
        "mood": selected_mood,
        "movies": results
    })


# ---------------------------------------------------------
# START SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\nCineMood AI is starting...")
    print("Open http://127.0.0.1:5000 in your browser.\n")
    app.run(debug=True)
