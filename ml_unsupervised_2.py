import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

movies_uri = 'https://raw.githubusercontent.com/oyurimatheus/clusterirng/master/movies/movies.csv'
movies = pd.read_csv(movies_uri)

genres = movies.genres.str.get_dummies()

scaler = StandardScaler()
scaled_genres = scaler.fit_transform(genres)

movies = pd.concat([movies, genres], axis=1) #concat data horizontaly

model = KMeans(n_clusters=17)
model.fit(scaled_genres)

print(f'Clusters Generated: {model.labels_}')
print(f'Columns: {genres.columns}')
print(f'\n\nCentroids Distribution:')

index = 0
results = list()
for centroids in model.cluster_centers_:
	index += 1
	relevant_genrers = list()
	if index == 1:
		print(f'Cluster number {index}:')
	else:
		print(f'\n\nCluster number {index}:')
	for inner_index in range(0, 19):
		print(f'{genres.columns[inner_index]}: {centroids[inner_index]}')
		inner_index += 1
		if centroids[inner_index] > 0:
			relevant_genrers.append(genres.columns[inner_index])
	results.append(relevant_genrers)

index = 0
for result in results:
	index += 1
	print(f'\n\nResults for Cluster number {index}:')
	for genrer in result:
		print(genrer)