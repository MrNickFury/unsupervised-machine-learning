import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

movies_uri = 'https://raw.githubusercontent.com/oyurimatheus/clusterirng/master/movies/movies.csv'
movies = pd.read_csv(movies_uri)

genres = movies.genres.str.get_dummies()

scaler = StandardScaler()
scaled_genres = scaler.fit_transform(genres)

movies = pd.concat([movies, genres], axis=1) #concat data horizontaly

clusters_quantity = 17

model = KMeans(n_clusters=clusters_quantity)
model.fit(scaled_genres)

groups = pd.DataFrame(model.cluster_centers_, columns=genres.columns)

#transpose the dataframe matrix turning its rows in columns and vice versa, so that we are able to visualize
#it from the perspective we want
groups.transpose()

distance_matrix = linkage(groups)

dendrogram = dendrogram(distance_matrix) #the dendrogram shows in a tree shape the distance between the categories
dendrogram

plt.show()
