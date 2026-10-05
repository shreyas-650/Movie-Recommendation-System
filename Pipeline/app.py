import pandas as pd
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies = pd.read_csv('./tmdb_5000_movies.csv')
credits = pd.read_csv('./tmdb_5000_credits.csv')
temp = movies.merge(credits,on='title')
df = temp[['id','title','overview','genres','keywords','release_date','cast','crew']]

#------------ Data Cleaning

#------------ Extraction Function -----------
def name_extract(col):
    temp = []
    for i in ast.literal_eval(col):
        temp.append(i['name'])
    return temp

def cast_extract(col):
    temp =[]
    j=0
    for i in ast.literal_eval(col):
        if j<3:
            temp.append(i['name'])
            j+=1
    return temp

def crew_extract(col):
    temp =[]
    for i in ast.literal_eval(col):
        if i['job'] == 'Director':
            temp.append(i['name'])
            break
    return temp

df['release_date']= pd.to_datetime(df['release_date'])
df['release_date'] = df['release_date'].dt.year.astype('Int16')

df['genres']=df['genres'].apply(name_extract)
df['keywords'] = df['keywords'].apply(name_extract)
df['cast'] = df['cast'].apply(cast_extract)
df['crew'] = df['crew'].apply(crew_extract)

df.dropna(inplace=True)
df['overview'] = df['overview'].apply(lambda x:x.split()) 

#---------------      Space Removed

space_rem = ['cast','crew','keywords','genres']
for x in space_rem:
    df[x] =df[x].apply(lambda x:[i.replace(' ', '') for i in x])
df['release_date'] = df['release_date'].apply(lambda x: [str(x)])
#--------------------- Creating Tags
df['tags'] = df['genres'] + df['keywords'] + df['cast'] + df['crew'] + df['release_date'] + df['overview'] 
df.drop(columns=['genres','overview','cast','keywords','crew','release_date'],inplace=True)
df['tags'] = df['tags'].apply(lambda x: " ".join(x))
df['tags'] = df['tags'].apply(lambda x: x.lower())
df['title'] = df['title'].apply(lambda x: x.lower())

#------------------- Stremming
ps = PorterStemmer()

def stemming(test):
    temp=[]
    for i in test.split():
        temp.append(ps.stem(i))
    return " ".join(temp)

df['tags'] = df['tags'].apply(stemming)


#--------------------  Feature Engineering{Vectorization}
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(df['tags'])
vectors = vectors.toarray()


#--------------------   cosine_similarity
similarities = cosine_similarity(vectors)

def recommend(movie):
    movie_index = df[df['title']==movie].index[0]
    recommendation = similarities[movie_index]
    movie_list = sorted(enumerate(recommendation),reverse=True,key=lambda x: x[1])[1:6]
    for i in movie_list:
        print(df.iloc[i[0]].title)

#---------------------   Movie List
recommend('avatar')

import pickle

with open('movies.pickle','wb') as movies:
    pickle.dump(df,movies)

import joblib
joblib.dump(similarities,'similarity.joblib',compress=3)