import streamlit as st
import joblib
import pandas as pd


knn_model = joblib.load('./pickles/rec_knn_model.joblib')
mlb_model = joblib.load('./pickles/mlb_genre_encoder.joblib')

movies_df =  pd.read_json("./data/movie_data.json")


# Load models + data
knn_model = joblib.load('./pickles/rec_knn_model.joblib')
mlb_model = joblib.load('./pickles/mlb_genre_encoder.joblib')
movies_df = pd.read_json("./data/movie_data.json")

# 🔹 Function to get recommendations
def get_recommendations(genres: list, k=6):
    genres_vec = mlb_model.transform([genres])
    distances, indices = knn_model.kneighbors(genres_vec, n_neighbors=k)
    rec_movies = []

    for i in range(1, len(distances.flatten())):
        idx = indices.flatten()[i]
        movie_id = int(idx) 
        sim_score = 1 - distances.flatten()[i]  # cosine similarity = 1 - distance
        current_movie = movies_df.loc[movie_id].to_dict()
        current_movie["similarity"] = round(sim_score * 100, 2)
        rec_movies.append(current_movie)

    return sorted(rec_movies, key=lambda x: x['similarity'], reverse=True)
     


# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.title("🎥 Movie Recommender System")
st.write("Find movies you’ll love – based on **genre** or a **movie you watched**.")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
rec_type = st.sidebar.radio("Recommendation Type", ["By Genre", "By Movie Watched"])
num_recs = st.sidebar.slider("Number of Recommendations", min_value=3, max_value=15, value=6, step=1)

# 🔹 Custom CSS for movie cards
st.markdown("""
    <style>
    .movie-card {
        background-color: #1e1e2f;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 18px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
        color: white;
    }
    .movie-title {
        font-size: 20px;
        font-weight: bold;
        color: #f4c10f;
    }
    .movie-info {
        font-size: 14px;
        margin: 6px 0;
    }
    .movie-link a {
        color: #58a6ff;
        text-decoration: none;
    }
    </style>
""", unsafe_allow_html=True)

def show_movie_card(movie):
    st.markdown(f"""
        <div class="movie-card">
            <div class="movie-title">🎬 {movie['movie_title']}</div>
            <div class="movie-info">📅 Year: {int(movie.get('title_year'))if movie.get('title_year') else 'N/A'} | 🌐 Language: {movie.get('language', 'N/A')} | 🌍 Country: {movie.get('country', 'N/A')}</div>
            <div class="movie-info">🎭 Genres: {', '.join(movie['genres'])}</div>
            <div class="movie-info">🎬 Director: {movie.get('director_name', 'N/A')}</div>
            <div class="movie-info">⏱️ Duration: {movie.get('duration', 'N/A')} min</div>
            <div class="movie-info">⭐ IMDb Score: {movie.get('imdb_score', 'N/A')}</div>
            <div class="movie-info">📊 Similarity: {movie.get('similarity', 0)}%</div>
            <div class="movie-link">🔗 <a href="{movie.get('movie_imdb_link', '#')}" target="_blank">View on IMDb</a></div>
        </div>
    """, unsafe_allow_html=True)

# 🔹 Recommendation by Genre
if rec_type == "By Genre":
    all_genres = sorted(set(g for genres in movies_df['genres'] for g in genres))
    chosen_genres = st.multiselect("🎭 Choose genres", all_genres)
    print(chosen_genres, num_recs)

    if st.button("Recommend Movies"):
        try :
            if chosen_genres:
                if len(chosen_genres) <= 1 : 
                    num_recs = 6
                    st.warning("Please select another genre to get more results.")

                recs = get_recommendations(chosen_genres, k=num_recs)
                st.subheader(f"🎯 Top {num_recs} recommendations for {', '.join(chosen_genres)}:")

                for movie in recs:
                    show_movie_card(movie)
            else:
                st.warning("Please select at least one genre.")
        except Exception as e: 
            print("eeror", e)

# 🔹 Recommendation by Movie
else:
    movie_choice = st.selectbox("🎞️ Pick a movie you watched", movies_df["movie_title"].tolist())

    if st.button("Recommend Similar Movies"):
        # Get genres of the chosen movie
        chosen_genres = movies_df[movies_df["movie_title"] == movie_choice]["genres"].iloc[0]
        recs = get_recommendations(chosen_genres, k=num_recs+1)

        st.subheader(f"🎯 Top {num_recs} recommendations similar to **{movie_choice}**:")

        for movie in recs:
            show_movie_card(movie)



