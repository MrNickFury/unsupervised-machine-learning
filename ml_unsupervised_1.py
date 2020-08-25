import pandas as pd
from sklearn.preprocessing import StandardScaler

movies_uri = 'https://raw.githubusercontent.com/oyurimatheus/clusterirng/master/movies/movies.csv'
movies = pd.read_csv(movies_uri)

genres = movies.genres.str.get_dummies()

scaler = StandardScaler()
scaled_genres = scaler.fit_transform(genres)

movies = pd.concat([movies, genres], axis=1) #concat data horizontaly

print('MOVIES\n', movies.head())
print('SCALED GENRES\n', scaled_genres)