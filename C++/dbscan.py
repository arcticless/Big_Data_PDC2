#!/usr/bin/env python


import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth,DBSCAN
from sklearn.datasets.samples_generator import make_blobs
import csv

mycsv = []
with open('MeanshiftInput.csv', 'r') as csvfile :
    spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
    for row in spamreader :
        row1 = int(row[0])
        row2 = int(row[1])
        mycsv.append([row1, row2])


X = np.array(mycsv, np.int)
###############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
db = DBSCAN(eps=1, min_samples=10).fit(X)
labels = db.labels_
cluster_for_each_point =  labels
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)


out = []
for i, point in enumerate(mycsv) :
    out.append([cluster_for_each_point[i], point[0], point[1]])


with open('MeanshiftOutput.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|')
    for elem in enumerate(out) :
        row = [elem[1][0] , elem[1][1], elem[1][2] ]
        spamwriter.writerow(row)



##graphic
# import pylab as pl
# from itertools import cycle

# pl.figure(1)
# pl.clf()

# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     my_members = labels == k
#     cluster_center = cluster_centers[k]
#     pl.plot(X[my_members, 0], X[my_members, 1], col + '.')
#     pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#                                     markeredgecolor='k', markersize=14)
# pl.title('Estimated number of clusters: %d' % n_clusters_)
# pl.show()

