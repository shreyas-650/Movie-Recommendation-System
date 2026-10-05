import streamlit as st
import pickle
import joblib
import pandas as pd
import sklearn
import nltk

st.title('Movie Recommendation')

with open('movies.pickle','rb') as m:
    df = pickle.load(m)
similarity = joblib.load('similarity.joblib','wb')

movies_name = df['title'].values
def recommend(movie_name):
    movie_index = df[df['title']==movie_name].index[0]
    recommendation = similarity[movie_index]
    movie_list = sorted(enumerate(recommendation),reverse=True,key=lambda x: x[1])[1:6]
    final = []
    for i in movie_list:
        final.append(df.iloc[i[0]].title)
    return final



movie_name = st.selectbox('Select Movie',movies_name)




if st.button('Recommend'):
    r = recommend(movie_name)
    st.write('Recommended Movies are:')
    for i in r:
        st.write(i)