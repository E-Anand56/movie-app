# app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Title
st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on your favorite movie!")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel('movie_data_with_year.xlsx')
    df.fillna('', inplace=True)
    df = df[df['Title'] != '']
    df['combined_features'] = df['Top 3 Genres'] + ' ' + df['Top 5 Cast'] + ' ' + df['Title']
    df['Title_lower'] = df['Title'].str.lower()
    return df

df = load_data()

# TF-IDF + cosine similarity
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['Title_lower']).drop_duplicates()

# Recommendation function
def recommend_movies(user_input, num_recommendations=5):
    user_input = user_input.lower().strip()
    matched_title, score = process.extractOne(user_input, df['Title_lower'].tolist())
    
    if score < 60:
        return None, "❌ Movie not found in dataset. Please try another title."
    
    idx = df[df['Title_lower'] == matched_title].index[0]
    original_title = df.loc[idx, 'Title']
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = df['Title'].iloc[movie_indices].tolist()
    return original_title, recommendations

# Streamlit input
user_input = st.text_input("🎬 Enter a movie name")
if st.button("Get Recommendations"):
    if user_input:
        original_title, result = recommend_movies(user_input)
        if result is None:
            st.error("❌ Movie not found. Try a different title.")
        else:
            st.success(f"✅ Showing results for: {original_title}")
            st.subheader("🎯 Top 5 Recommended Movies:")
            for i, title in enumerate(result, start=1):
                st.write(f"{i}. {title}")
    else:
        st.warning("⚠️ Please enter a movie name.")
