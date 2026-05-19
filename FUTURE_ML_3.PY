import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
resumes = pd.read_csv("D:\FUTURE INTERN\DATASET 3.txt")
job_desc = pd.read_csv("D:\FUTURE INTERN\DATASET 3.1.txt").iloc[0,0]

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(resumes['resume'].tolist() + [job_desc])

# Compute similarity
similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

# Ranking
resumes['similarity'] = similarities
print(resumes.sort_values(by='similarity', ascending=False))
