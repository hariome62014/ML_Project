
# Content-Based Movie Recommender System Using Cosine Similarity

## Overview
This project is a content-based movie recommender system that suggests movies to users based on their similarity to a given movie. The similarity is calculated using cosine similarity, which compares movies based on their features, such as genre, director, and cast.

## Project Structure
1. **Data Collection and Preprocessing**: The dataset is collected and preprocessed to extract relevant features.
2. **Feature Extraction**: Movie features are converted into vectors using techniques like TF-IDF.
3. **Cosine Similarity Calculation**: Cosine similarity is computed between the movie vectors.
4. **Recommendation Generation**: A list of similar movies is generated and presented to the user.
5. **Evaluation**: The system's performance is evaluated using appropriate metrics.

## Implementation
### 1. Import Libraries
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

### 2. Load Dataset
```python
df = pd.read_csv('movies_dataset.csv')
```

### 3. Data Preprocessing
```python
tfidf = TfidfVectorizer(stop_words='english')
df['genre'] = df['genre'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['genre'])
```

### 4. Calculate Cosine Similarity
```python
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

### 5. Build Recommender Function
```python
def recommend_movies(title, cosine_sim=cosine_sim):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]
```

### 6. Test the Recommender
```python
recommended_movies = recommend_movies('The Dark Knight')
print("Recommended Movies:")
print(recommended_movies)
```

## Conclusion
This content-based recommender system provides movie recommendations based on cosine similarity of movie features. It is a fundamental approach in recommendation systems, emphasizing the use of content information over collaborative filtering methods.

## Requirements
- Python 3.x
- Pandas
- Scikit-learn
- NumPy

## How to Run
1. Clone the repository.
2. Install the required libraries.
3. Run the `main.py` file to test the recommender system.
