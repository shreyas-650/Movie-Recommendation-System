# 🎬 Movie Recommendation System

A content-based movie recommendation system that recommends movies based on their similarity to a selected movie.

## 📌 Features

- Movie data processing using Pandas
- Extraction of genres, keywords, cast and director
- Release year extraction
- Text preprocessing and stemming using NLTK
- Feature extraction using Bag of Words
- Movie similarity calculation using Cosine Similarity
- Top 5 similar movie recommendations

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Streamlit

## 🔄 Workflow

```text
Movie Dataset
      ↓
Data Cleaning
      ↓
Feature Extraction
      ↓
Create Movie Tags
      ↓
Text Preprocessing
      ↓
Stemming
      ↓
Bag of Words
      ↓
Cosine Similarity
      ↓
Movie Recommendations
```

## 🧠 Recommendation Approach

This project uses a **content-based recommendation approach**.

Movie information such as:

- Genres
- Keywords
- Top 3 Cast Members
- Director
- Release Year
- Movie Overview

is combined into a single `tags` feature.

The tags are converted into numerical vectors using **CountVectorizer (Bag of Words)**.

Cosine similarity is then used to measure the similarity between movies and recommend the most similar movies.

## 📊 Dataset

The project uses the **TMDB 5000 Movies Dataset** and **TMDB 5000 Credits Dataset**.

## 🚀 Example

```python
recommend('avatar')
```

The system returns the **top 5 movies similar to Avatar**.

## 📁 Project Structure

```text
Movie-Recommendation-System/
│
├── app.py
├── notebook.ipynb
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
├── movies.pickle
├── similarity.joblib
├── requirements.txt
└── README.md
```

## 📦 Installation

Clone the repository and install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 🎯 Project Objective

To build a content-based movie recommendation system that identifies and recommends movies with similar characteristics using NLP-based text processing and cosine similarity.