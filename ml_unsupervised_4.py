import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

movies_uri = 'https://raw.githubusercontent.com/oyurimatheus/clusterirng/master/movies/movies.csv'
movies = pd.read_csv(movies_uri)

genres = movies.genres.str.get_dummies()

scaler = StandardScaler()
scaled_genres = scaler.fit_transform(genres)

movies = pd.concat([movies, genres], axis=1)  # concat data horizontaly

def kmeans(clusters_quantity, genres):
    model = KMeans(n_clusters=clusters_quantity)
    model.fit(genres)
    """ inertia returns the calculated error for that dataset clustering, KMeans uses mean squarred error to measure
    how much should the result be penalized for being to far away from its centroid, nonetheless when analizing
    its results in a graphic we make a decision based on the elbow method, once the lower the results the closer
    the data are from its centroid without messing up with too many centroids """
    return [clusters_quantity, model.inertia_]


results = [kmeans(number_of_groups, scaled_genres)
           for number_of_groups in range(1, 18)]
results = pd.DataFrame(results, columns=['groups', 'inertia'])

results.inertia.plot(xticks=results.groups)
plt.show()
