import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Page title
st.set_page_config(page_title="🎬 Movie Recommendation App", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Enter your favorite movie to get similar movie recommendations.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel('movie_data_with_year.xlsx')
    df.fillna('', inplace=True)
    df = df[df['Title'] != '']
    df['combined_features'] = df['Top 3 Genres'] + ' ' + df['Top 5 Cast'] + ' ' + df['Title']
    df['Title_lower'] = df['Title'].str.lower()
    return df

df = load_data()

# TF-IDF and cosine similarity
@st.cache_resource
def compute_similarity(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = compute_similarity(df)

# Recommend function
def recommend_movies(user_input, num_recommendations=5):
    user_input = user_input.lower().strip()
    matched_title, score = process.extractOne(user_input, df['Title_lower'].tolist())

    if score < 80:
        return None, []

    idx = df[df['Title_lower'] == matched_title].index[0]
    original_title = df.loc[idx, 'Title']

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]

    return original_title, df['Title'].iloc[movie_indices].tolist()

# User input
user_movie = st.text_input("🎥 Enter a movie name:")

if st.button("🔍 Recommend"):
    if user_movie:
        matched_title, recommendations = recommend_movies(user_movie)

        if recommendations:
            st.success(f"✅ Showing results for: {matched_title}")
            st.markdown("### 🎯 Top 5 Similar Movies:")
            for i, movie in enumerate(recommendations, start=1):
                st.write(f"{i}. {movie}")
        else:
            st.error("❌ Movie not found in dataset. Please check the spelling or try another title.")
