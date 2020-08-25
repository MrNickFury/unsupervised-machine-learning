import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

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

tsne = TSNE() #an algorithm to reduce the 20 dimensions we are analizing to 2 dimensions that we can visualize, important to remember that when we reduce dimensions we always lose information
visualization = tsne.fit_transform(scaled_genres)

sns.set(rc={'figure.figsize': (10, 10)})
sns.scatterplot(
	x=visualization[:, 0], 
	y=visualization[:, 1], 
	hue=model.labels_, 
	palette=sns.color_palette("Set1", clusters_quantity)
)

plt.show()
