# ============================================================
# PR12 AI/ML PROJECT
# Mood and Genre Based Movie Recommendation System
# ============================================================

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import tkinter as tk
from tkinter import ttk, messagebox


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading dataset...")

try:
    movies = pd.read_csv("movies.csv")
except FileNotFoundError:
    print("ERROR: movies.csv not found.")
    print("Please keep movies.csv in the same folder as main.py")
    input("\nPress Enter to exit...")
    exit()

print("Dataset loaded successfully.")
print("Original dataset shape:", movies.shape)


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

movies = movies.dropna(
    subset=["title", "genres"]
)

movies = movies.drop_duplicates(
    subset=["title"]
).reset_index(drop=True)

movies["genres"] = movies["genres"].replace(
    "(no genres listed)",
    ""
)

movies["genres"] = movies["genres"].fillna("")

movies["genres_clean"] = movies["genres"].str.replace(
    "|",
    " ",
    regex=False
)


# ============================================================
# 3. MOOD MAPPING
# ============================================================
#
# MovieLens does not contain a mood column.
# Therefore, moods are inferred from movie genres.
#
# Example:
# Action -> Excited
# Comedy -> Happy
# Horror -> Scared
# Romance -> Romantic
# ============================================================

MOOD_MAPPING = {

    "Happy": [
        "Comedy",
        "Animation",
        "Children",
        "Musical"
    ],

    "Romantic": [
        "Romance",
        "Comedy"
    ],

    "Excited": [
        "Action",
        "Adventure",
        "Thriller",
        "Sci-Fi"
    ],

    "Suspenseful": [
        "Thriller",
        "Mystery",
        "Horror",
        "Crime"
    ],

    "Scared": [
        "Horror",
        "Thriller"
    ],

    "Emotional": [
        "Drama",
        "Romance"
    ],

    "Relaxed": [
        "Animation",
        "Comedy",
        "Children",
        "Musical"
    ],

    "Curious": [
        "Documentary",
        "Mystery",
        "Sci-Fi"
    ],

    "Serious": [
        "Drama",
        "Documentary",
        "Crime"
    ],

    "Adventurous": [
        "Adventure",
        "Action",
        "Fantasy"
    ]
}


# ============================================================
# 4. FIND MOODS FOR EACH MOVIE
# ============================================================

def get_movie_moods(genres):

    movie_genres = set(
        genres.split("|")
    )

    moods = []

    for mood, mood_genres in MOOD_MAPPING.items():

        for genre in mood_genres:

            if genre in movie_genres:
                moods.append(mood)
                break

    return moods


movies["moods"] = movies["genres"].apply(
    get_movie_moods
)


# ============================================================
# 5. TF-IDF MODEL
# ============================================================

print("Building TF-IDF model...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    movies["genres_clean"]
)

print(
    "TF-IDF model created successfully."
)

print(
    "TF-IDF matrix shape:",
    tfidf_matrix.shape
)


# ============================================================
# 6. AVAILABLE GENRES
# ============================================================

available_genres = sorted(
    set(
        genre
        for genre_list in movies["genres"]
        for genre in genre_list.split("|")
        if genre
    )
)


# ============================================================
# 7. GENRE MATCHING
# ============================================================

def calculate_genre_match(
    movie_genres,
    selected_genre
):

    if not selected_genre:
        return 0.0

    movie_genre_set = set(
        movie_genres.split("|")
    )

    if selected_genre in movie_genre_set:
        return 1.0

    return 0.0


# ============================================================
# 8. MOOD MATCHING
# ============================================================

def calculate_mood_match(
    movie_moods,
    selected_mood
):

    if not selected_mood:
        return 0.0

    if selected_mood in movie_moods:
        return 1.0

    return 0.0


# ============================================================
# 9. TF-IDF COSINE SIMILARITY
# ============================================================

def calculate_content_similarity(
    selected_genre
):

    if not selected_genre:
        return np.zeros(
            len(movies)
        )

    query_vector = vectorizer.transform(
        [selected_genre]
    )

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    return similarities


# ============================================================
# 10. MAIN RECOMMENDATION FUNCTION
# ============================================================

def recommend_movies(
    selected_genre,
    selected_mood,
    number=5,
    genre_weight=0.5,
    mood_weight=0.3,
    similarity_weight=0.2
):

    # --------------------------------------------------------
    # Calculate TF-IDF similarity
    # --------------------------------------------------------

    similarity_scores = calculate_content_similarity(
        selected_genre
    )

    final_scores = []

    # --------------------------------------------------------
    # Calculate final score for every movie
    # --------------------------------------------------------

    for idx in range(len(movies)):

        genre_score = calculate_genre_match(
            movies.loc[idx, "genres"],
            selected_genre
        )

        mood_score = calculate_mood_match(
            movies.loc[idx, "moods"],
            selected_mood
        )

        similarity_score = similarity_scores[idx]

        # Final weighted score
        final_score = (
            genre_weight * genre_score
            +
            mood_weight * mood_score
            +
            similarity_weight * similarity_score
        )

        final_scores.append(
            final_score
        )

    # --------------------------------------------------------
    # Create result dataframe
    # --------------------------------------------------------

    result = movies.copy()

    result["recommendation_score"] = (
        final_scores
    )

    result["similarity_score"] = (
        similarity_scores
    )

    # --------------------------------------------------------
    # Sort by recommendation score
    # --------------------------------------------------------

    result = result.sort_values(
        by="recommendation_score",
        ascending=False
    )

    # --------------------------------------------------------
    # Return top N movies
    # --------------------------------------------------------

    recommendations = result.head(
        number
    )

    return recommendations


# ============================================================
# 11. DISPLAY RECOMMENDATIONS
# ============================================================

def display_recommendations(
    recommendations
):

    print("\n====================================")
    print("RECOMMENDED MOVIES")
    print("====================================")

    for rank, (_, movie) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        print(
            f"\n{rank}. {movie['title']}"
        )

        print(
            f"   Genres: {movie['genres']}"
        )

        if movie["moods"]:

            print(
                "   Moods: "
                + ", ".join(movie["moods"])
            )

        else:

            print(
                "   Moods: Not available"
            )

        print(
            f"   Recommendation Score: "
            f"{movie['recommendation_score']:.3f}"
        )


# ============================================================
# 12. PRECISION@5 EVALUATION
# ============================================================

def evaluate_model(
    genre_weight,
    mood_weight,
    similarity_weight,
    sample_size=100
):

    rng = np.random.RandomState(
        42
    )

    actual_sample_size = min(
        sample_size,
        len(movies)
    )

    sample_indices = rng.choice(
        len(movies),
        size=actual_sample_size,
        replace=False
    )

    precision_scores = []

    for idx in sample_indices:

        movie_genres = movies.loc[
            idx,
            "genres"
        ]

        movie_moods = movies.loc[
            idx,
            "moods"
        ]

        # Skip movies without genre
        if not movie_genres:
            continue

        # Choose first genre
        genres = movie_genres.split("|")

        selected_genre = genres[0]

        # Choose first mood
        if movie_moods:

            selected_mood = movie_moods[0]

        else:

            selected_mood = ""

        # Get recommendations
        recommendations = recommend_movies(
            selected_genre,
            selected_mood,
            number=5,
            genre_weight=genre_weight,
            mood_weight=mood_weight,
            similarity_weight=similarity_weight
        )

        relevant_count = 0

        for _, recommended_movie in recommendations.iterrows():

            recommended_genres = set(
                recommended_movie["genres"].split("|")
            )

            recommended_moods = set(
                recommended_movie["moods"]
            )

            genre_match = (
                selected_genre
                in recommended_genres
            )

            mood_match = (
                selected_mood
                in recommended_moods
            )

            if genre_match or mood_match:

                relevant_count += 1

        # Precision@5
        precision = (
            relevant_count / 5
        )

        precision_scores.append(
            precision
        )

    if precision_scores:

        return float(
            np.mean(precision_scores)
        )

    return 0.0


# ============================================================
# 13. MODEL OPTIMIZATION
# ============================================================

print("\n====================================")
print("MODEL OPTIMIZATION")
print("====================================")

print(
    "Testing different recommendation weights..."
)

print(
    "Please wait..."
)


# ------------------------------------------------------------
# MODEL 1
# Genre = 0.5
# Mood = 0.3
# Similarity = 0.2
# ------------------------------------------------------------

score_model_1 = evaluate_model(
    genre_weight=0.5,
    mood_weight=0.3,
    similarity_weight=0.2,
    sample_size=100
)

print(
    "\nModel 1"
)

print(
    "Genre Weight = 0.5"
)

print(
    "Mood Weight = 0.3"
)

print(
    "Similarity Weight = 0.2"
)

print(
    "Precision@5 =",
    round(score_model_1, 4)
)


# ------------------------------------------------------------
# MODEL 2
# Genre = 0.4
# Mood = 0.4
# Similarity = 0.2
# ------------------------------------------------------------

score_model_2 = evaluate_model(
    genre_weight=0.4,
    mood_weight=0.4,
    similarity_weight=0.2,
    sample_size=100
)

print(
    "\nModel 2"
)

print(
    "Genre Weight = 0.4"
)

print(
    "Mood Weight = 0.4"
)

print(
    "Similarity Weight = 0.2"
)

print(
    "Precision@5 =",
    round(score_model_2, 4)
)


# ------------------------------------------------------------
# MODEL 3
# Genre = 0.6
# Mood = 0.2
# Similarity = 0.2
# ------------------------------------------------------------

score_model_3 = evaluate_model(
    genre_weight=0.6,
    mood_weight=0.2,
    similarity_weight=0.2,
    sample_size=100
)

print(
    "\nModel 3"
)

print(
    "Genre Weight = 0.6"
)

print(
    "Mood Weight = 0.2"
)

print(
    "Similarity Weight = 0.2"
)

print(
    "Precision@5 =",
    round(score_model_3, 4)
)


# ============================================================
# 14. SELECT BEST MODEL
# ============================================================

model_scores = {

    "Model 1": score_model_1,

    "Model 2": score_model_2,

    "Model 3": score_model_3
}


best_model = max(
    model_scores,
    key=model_scores.get
)

best_score = model_scores[
    best_model
]


print("\n====================================")
print("OPTIMIZATION RESULT")
print("====================================")

print(
    "Best Model:",
    best_model
)

print(
    "Best Precision@5:",
    round(best_score, 4)
)


# ============================================================
# 15. BEST MODEL WEIGHTS
# ============================================================

if best_model == "Model 1":

    BEST_GENRE_WEIGHT = 0.5
    BEST_MOOD_WEIGHT = 0.3
    BEST_SIMILARITY_WEIGHT = 0.2

elif best_model == "Model 2":

    BEST_GENRE_WEIGHT = 0.4
    BEST_MOOD_WEIGHT = 0.4
    BEST_SIMILARITY_WEIGHT = 0.2

else:

    BEST_GENRE_WEIGHT = 0.6
    BEST_MOOD_WEIGHT = 0.2
    BEST_SIMILARITY_WEIGHT = 0.2


# ============================================================
# 16. COMMAND LINE RECOMMENDER
# ============================================================

def command_line_demo():

    print("\n====================================")
    print("COMMAND LINE RECOMMENDER")
    print("====================================")

    print("\nAvailable Genres:")

    print(
        ", ".join(
            available_genres
        )
    )

    print("\nAvailable Moods:")

    print(
        ", ".join(
            MOOD_MAPPING.keys()
        )
    )

    selected_genre = input(
        "\nEnter your preferred genre: "
    ).strip()

    selected_mood = input(
        "Enter your current mood: "
    ).strip()

    # Check genre
    if selected_genre not in available_genres:

        print(
            "\nInvalid genre."
        )

        return

    # Check mood
    if selected_mood not in MOOD_MAPPING:

        print(
            "\nInvalid mood."
        )

        return

    recommendations = recommend_movies(
        selected_genre,
        selected_mood,
        number=5,
        genre_weight=BEST_GENRE_WEIGHT,
        mood_weight=BEST_MOOD_WEIGHT,
        similarity_weight=BEST_SIMILARITY_WEIGHT
    )

    display_recommendations(
        recommendations
    )


# ============================================================
# 17. TKINTER GUI
# ============================================================

def create_gui():

    window = tk.Tk()

    window.title(
        "Mood & Genre Movie Recommendation System"
    )

    window.geometry(
        "800x700"
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    title = tk.Label(
        window,
        text="Movie Recommendation System",
        font=("Arial", 22, "bold")
    )

    title.pack(
        pady=20
    )

    subtitle = tk.Label(
        window,
        text="Recommendation based on your Genre and Mood",
        font=("Arial", 12)
    )

    subtitle.pack(
        pady=5
    )

    # --------------------------------------------------------
    # GENRE
    # --------------------------------------------------------

    genre_label = tk.Label(
        window,
        text="Select Genre:",
        font=("Arial", 12, "bold")
    )

    genre_label.pack(
        pady=(20, 5)
    )

    genre_combo = ttk.Combobox(
        window,
        values=available_genres,
        state="readonly",
        width=35
    )

    genre_combo.pack()

    genre_combo.set(
        "Comedy"
    )

    # --------------------------------------------------------
    # MOOD
    # --------------------------------------------------------

    mood_label = tk.Label(
        window,
        text="Select Mood:",
        font=("Arial", 12, "bold")
    )

    mood_label.pack(
        pady=(20, 5)
    )

    mood_combo = ttk.Combobox(
        window,
        values=list(
            MOOD_MAPPING.keys()
        ),
        state="readonly",
        width=35
    )

    mood_combo.pack()

    mood_combo.set(
        "Happy"
    )

    # --------------------------------------------------------
    # RESULT BOX
    # --------------------------------------------------------

    result_box = tk.Text(
        window,
        height=20,
        width=90,
        font=("Arial", 10)
    )

    result_box.pack(
        pady=20
    )

    # --------------------------------------------------------
    # BUTTON FUNCTION
    # --------------------------------------------------------

    def get_recommendations():

        selected_genre = genre_combo.get()

        selected_mood = mood_combo.get()

        if not selected_genre:

            messagebox.showwarning(
                "Missing Genre",
                "Please select a genre."
            )

            return

        if not selected_mood:

            messagebox.showwarning(
                "Missing Mood",
                "Please select a mood."
            )

            return

        # Get recommendations
        recommendations = recommend_movies(
            selected_genre,
            selected_mood,
            number=5,
            genre_weight=BEST_GENRE_WEIGHT,
            mood_weight=BEST_MOOD_WEIGHT,
            similarity_weight=BEST_SIMILARITY_WEIGHT
        )

        # Clear old results
        result_box.delete(
            "1.0",
            tk.END
        )

        # Heading
        result_box.insert(
            tk.END,
            "RECOMMENDED MOVIES\n"
        )

        result_box.insert(
            tk.END,
            "==============================\n\n"
        )

        result_box.insert(
            tk.END,
            f"Selected Genre: "
            f"{selected_genre}\n"
        )

        result_box.insert(
            tk.END,
            f"Selected Mood: "
            f"{selected_mood}\n\n"
        )

        # Display movies
        for rank, (_, movie) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            result_box.insert(
                tk.END,
                f"{rank}. {movie['title']}\n"
            )

            result_box.insert(
                tk.END,
                f"   Genres: "
                f"{movie['genres']}\n"
            )

            if movie["moods"]:

                result_box.insert(
                    tk.END,
                    "   Moods: "
                    + ", ".join(
                        movie["moods"]
                    )
                    + "\n"
                )

            else:

                result_box.insert(
                    tk.END,
                    "   Moods: Not available\n"
                )

            result_box.insert(
                tk.END,
                f"   Score: "
                f"{movie['recommendation_score']:.3f}\n\n"
            )

    # --------------------------------------------------------
    # RECOMMEND BUTTON
    # --------------------------------------------------------

    recommend_button = tk.Button(
        window,
        text="Recommend Movies",
        command=get_recommendations,
        font=("Arial", 12, "bold"),
        padx=25,
        pady=10
    )

    recommend_button.pack(
        pady=10
    )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    footer = tk.Label(
        window,
        text="PR12 AI/ML Project",
        font=("Arial", 9)
    )

    footer.pack(
        side="bottom",
        pady=10
    )

    # --------------------------------------------------------
    # START GUI
    # --------------------------------------------------------

    window.mainloop()


# ============================================================
# 18. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n============================================")
    print("MOOD & GENRE BASED MOVIE RECOMMENDATION")
    print("============================================")

    print(
        "\nModel optimization completed."
    )

    print(
        "Best model:",
        best_model
    )

    print(
        "Best Precision@5:",
        round(best_score, 4)
    )

    print(
        "\nStarting GUI..."
    )

    create_gui()