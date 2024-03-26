import streamlit as st
import pickle
import pandas as pd

def recomend(movie):
    movie_index=movies[movies['title'] == movie].index[0]
    distance=similarity[movie_index]
    movies_list=sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
    
    recomended_movies = []
    for i in movies_list:
        recomended_movies.append(movies.iloc[i[0]].title)
    
    return recomended_movies

movies_list = pickle.load(open('movie_dic.pkl','rb'))
movies = pd.DataFrame(movies_list)
movies_list= movies['title'].values

similarity = pickle.load(open('similarity.pkl','rb'))

print(movies_list)
st.title('Movie Recommender System')

option = st.selectbox(
    'What Movie you like?',
    movies_list
)

if st.button('Recommend'):
    recomdetions=recomend(option)
    for i in recomdetions:
        st.write(i)