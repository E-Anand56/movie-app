{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "e4f7e8ec-8f7f-40ed-91f4-a62e0b71274f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting fuzzywuzzy\n",
      "  Downloading fuzzywuzzy-0.18.0-py2.py3-none-any.whl.metadata (4.9 kB)\n",
      "Collecting python-Levenshtein\n",
      "  Downloading python_levenshtein-0.27.1-py3-none-any.whl.metadata (3.7 kB)\n",
      "Collecting Levenshtein==0.27.1 (from python-Levenshtein)\n",
      "  Downloading levenshtein-0.27.1-cp313-cp313-win_amd64.whl.metadata (3.6 kB)\n",
      "Collecting rapidfuzz<4.0.0,>=3.9.0 (from Levenshtein==0.27.1->python-Levenshtein)\n",
      "  Downloading rapidfuzz-3.13.0-cp313-cp313-win_amd64.whl.metadata (12 kB)\n",
      "Downloading fuzzywuzzy-0.18.0-py2.py3-none-any.whl (18 kB)\n",
      "Downloading python_levenshtein-0.27.1-py3-none-any.whl (9.4 kB)\n",
      "Downloading levenshtein-0.27.1-cp313-cp313-win_amd64.whl (100 kB)\n",
      "Downloading rapidfuzz-3.13.0-cp313-cp313-win_amd64.whl (1.6 MB)\n",
      "   ---------------------------------------- 0.0/1.6 MB ? eta -:--:--\n",
      "   ------ --------------------------------- 0.3/1.6 MB ? eta -:--:--\n",
      "   ------------------------- -------------- 1.0/1.6 MB 3.0 MB/s eta 0:00:01\n",
      "   -------------------------------- ------- 1.3/1.6 MB 2.7 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 1.6/1.6 MB 2.5 MB/s eta 0:00:00\n",
      "Installing collected packages: fuzzywuzzy, rapidfuzz, Levenshtein, python-Levenshtein\n",
      "Successfully installed Levenshtein-0.27.1 fuzzywuzzy-0.18.0 python-Levenshtein-0.27.1 rapidfuzz-3.13.0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "[notice] A new release of pip is available: 24.3.1 -> 25.1.1\n",
      "[notice] To update, run: python.exe -m pip install --upgrade pip\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pandas in c:\\users\\anand\\appdata\\local\\programs\\python\\python313\\lib\\site-packages (2.2.3)\n",
      "Requirement already satisfied: scikit-learn in c:\\users\\anand\\appdata\\local\\programs\\python\\python313\\lib\\site-packages (1.6.1)\n",
      "Requirement already satisfied: openpyxl in c:\\users\\anand\\appdata\\local\\programs\\python\\python313\\lib\\site-packages (3.1.5)\n",
      "Requirement already satisfied: numpy>=1.26.0 in c:\\users\\anand\\appdata\\local\\programs\\python\\python313\\lib\\site-packages (from pandas) (2.2.5)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\anand\\appdata\\local\\programs\\python\\python313\\lib\\site-packages (from pandas) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\anand\\appdata\\local\\programs\\python\\python313\\lib\\site-packages (from pandas) (2025.2)\n",
      "Requirement already satisfied: tzdata>=2022.7 in c:\\users\\anand\\appdata\\local\\programs\\python\\python313\\lib\\site-packages (from pandas) (2025.2)\n",
      "Requirement already satisfied: scipy>=1.6.0 in c:\\users\\anand\\appdata\\local\\programs\\python\\python313\\lib\\site-packages (from scikit-learn) (1.15.3)\n",
      "Requirement already satisfied: joblib>=1.2.0 in c:\\users\\anand\\appdata\\local\\programs\\python\\python313\\lib\\site-packages (from scikit-learn) (1.5.1)\n",
      "Requirement already satisfied: threadpoolctl>=3.1.0 in c:\\users\\anand\\appdata\\local\\programs\\python\\python313\\lib\\site-packages (from scikit-learn) (3.6.0)\n",
      "Requirement already satisfied: et-xmlfile in c:\\users\\anand\\appdata\\local\\programs\\python\\python313\\lib\\site-packages (from openpyxl) (2.0.0)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\anand\\appdata\\local\\programs\\python\\python313\\lib\\site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "[notice] A new release of pip is available: 24.3.1 -> 25.1.1\n",
      "[notice] To update, run: python.exe -m pip install --upgrade pip\n"
     ]
    }
   ],
   "source": [
    "!pip install fuzzywuzzy python-Levenshtein\n",
    "!pip install pandas scikit-learn openpyxl\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "e6b5ceb5-4bcc-462c-aa55-bc82eac03dd3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "🎬 Enter a movie name:  ms dhoni\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "✅ Showing results for: M.S. Dhoni: The Untold Story\n",
      "\n",
      "🎯 Top 5 movies similar to your search:\n",
      "1. Special 26\n",
      "2. Rang De Basanti\n",
      "3. OMG â€“ Oh My God!\n",
      "4. Dangal\n",
      "5. Dabangg\n"
     ]
    }
   ],
   "source": [
    "# 🎬 Content-Based Movie Recommendation System with Fuzzy Matching\n",
    "\n",
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from fuzzywuzzy import process\n",
    "\n",
    "# Step 1: Load dataset\n",
    "df = pd.read_excel('movie_data_with_year.xlsx')  # Your movie dataset file\n",
    "df.fillna('', inplace=True)\n",
    "df = df[df['Title'] != '']  # Remove rows with empty titles\n",
    "\n",
    "# Step 2: Combine features into a single string for each movie\n",
    "df['combined_features'] = df['Top 3 Genres'] + ' ' + df['Top 5 Cast'] + ' ' + df['Title']\n",
    "df['Title_lower'] = df['Title'].str.lower()\n",
    "\n",
    "# Step 3: Create TF-IDF matrix\n",
    "vectorizer = TfidfVectorizer(stop_words='english')\n",
    "tfidf_matrix = vectorizer.fit_transform(df['combined_features'])\n",
    "\n",
    "# Step 4: Compute cosine similarity between all movies\n",
    "cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)\n",
    "\n",
    "# Step 5: Create a title-to-index mapping\n",
    "indices = pd.Series(df.index, index=df['Title_lower']).drop_duplicates()\n",
    "\n",
    "# Step 6: Movie recommendation function\n",
    "def recommend_movies(user_input, num_recommendations=5):\n",
    "    user_input = user_input.lower().strip()\n",
    "    \n",
    "    # Fuzzy match the user input with titles\n",
    "    matched_title, score = process.extractOne(user_input, df['Title_lower'].tolist())\n",
    "    \n",
    "    if score < 60:\n",
    "        return \"❌ Movie not found in dataset. Please try another title.\"\n",
    "    \n",
    "    idx = df[df['Title_lower'] == matched_title].index[0]\n",
    "    original_title = df.loc[idx, 'Title']\n",
    "    \n",
    "    print(f\"\\n✅ Showing results for: {original_title}\")\n",
    "\n",
    "    # Get similarity scores\n",
    "    sim_scores = list(enumerate(cosine_sim[idx]))\n",
    "    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]\n",
    "    movie_indices = [i[0] for i in sim_scores]\n",
    "\n",
    "    return df['Title'].iloc[movie_indices].tolist()\n",
    "\n",
    "# Step 7: Get user input\n",
    "user_input = input(\"🎬 Enter a movie name: \")\n",
    "recommendations = recommend_movies(user_input)\n",
    "\n",
    "# Step 8: Display recommendations\n",
    "if isinstance(recommendations, list):\n",
    "    print(f\"\\n🎯 Top 5 movies similar to your search:\")\n",
    "    for i, title in enumerate(recommendations, start=1):\n",
    "        print(f\"{i}. {title}\")\n",
    "else:\n",
    "    print(recommendations)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
