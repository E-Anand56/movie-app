# 🎬 Streamlit Movie Recommendation App

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Load dataset
df = pd.read_excel('movie_data_with_year.xlsx')
df.fillna('', inplace=True)
df = df[df['Title'] != '']
df['combined_features'] = df['Top 3 Genres'] + ' ' + df['Top 5 Cast'] + ' ' + df['Title']
df['Title_lower'] = df['Title'].str.lower()

# TF-IDF and cosine similarity
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['Title_lower']).drop_duplicates()

# Streamlit UI
st.title("🎬 Movie Recommendation System")
user_input = st.text_input("Enter a movie name:")

def recommend_movies(user_input, num_recommendations=5):
    user_input = user_input.lower().strip()
    matched_title, score = process.extractOne(user_input, df['Title_lower'].tolist())

    if score < 60:
        return "❌ Movie not found. Try another title."

    idx = df[df['Title_lower'] == matched_title].index[0]
    original_title = df.loc[idx, 'Title']
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return original_title, df['Title'].iloc[movie_indices].tolist()

if user_input:
    result = recommend_movies(user_input)

    if isinstance(result, tuple):
        original_title, recommendations = result
        st.success(f"✅ Showing results for: {original_title}")
        st.markdown("### 🎯 Top 5 Similar Movies:")
        for i, movie in enumerate(recommendations, start=1):
            st.write(f"{i}. {movie}")
    else:
        st.error(result)
