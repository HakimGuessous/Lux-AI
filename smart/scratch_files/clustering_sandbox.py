import numpy as np
from sklearn.cluster import KMeans
import scipy


wood_locations = [(1,2), (2,2), (2,3), (5,5), (6,5)]
city_dist_matrix = np.zeros(shape=(12, 12))
city_dist_matrix[8][8] = 3
big_conv_matrix = np.array([[0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,1,1,1,0,0,0,0],
                            [0,0,0,1,1,1,1,1,0,0,0],
                            [0,0,1,1,1,1,1,1,1,0,0],
                            [0,1,1,1,1,1,1,1,1,1,0],
                            [1,1,1,1,1,1,1,1,1,1,1],
                            [0,1,1,1,1,1,1,1,1,1,0],
                            [0,0,1,1,1,1,1,1,1,0,0],
                            [0,0,0,1,1,1,1,1,0,0,0],
                            [0,0,0,0,1,1,1,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0]])

city_dist = scipy.signal.convolve2d(city_dist_matrix, big_conv_matrix, mode='same')
for x in range(12):
    for y in range(12):
        if city_dist[x][y] >= 3 and (x, y) in wood_locations:
            wood_locations.remove((x,y))

clusters = int(len(wood_locations)/8)
kmeans = KMeans(n_clusters=3, random_state=0).fit(wood_locations)
kmeans.labels_


