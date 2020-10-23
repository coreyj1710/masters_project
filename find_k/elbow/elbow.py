from sklearn.cluster import KMeans
import numpy as np


import sklearn.metrics as metrics
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

class Elbow:

    def __init__(self, results_folder):
        self.results_folder = results_folder

    def elbow_method(self, gene_df, iterations, highest_k):

        #time script
        startTime = datetime.now()
        
        print('\nApplying elbow method to search for best K...')

        M = len( gene_df )

        #-loop through various numbers of clusters
        k_val = []
        IN = [] # inertia (within cluster)
        DIS = [] # distortion

        #convert data to numpy array for distortion calculation
        genes_array = np.array(gene_df)

        #cluster and evaluate for each value of k
        for k in range(2, highest_k+1):
            print('Calculating for k =', k, 'of', highest_k)

            #add scores over multiple iterations and calculate average
            avg_IN = 0.0
            avg_DIS = 0.0

            for i in range(0, iterations):
                kmeans = KMeans(n_clusters=k, random_state=i).fit(gene_df)

                avg_IN += kmeans.inertia_
                avg_DIS += sum(np.min(distance.cdist(genes_array, kmeans.cluster_centers_, \
                      'euclidean'),axis=1)) / genes_array.shape[0]

            #average inertia score. Used for elbow method
            IN.append(avg_IN/iterations)
            #average distortion score. Used for elbow method
            DIS.append(avg_DIS/iterations)

            k_val.append(k)

        print('Time elapsed', datetime.now() - startTime)

        #create directory to save results
        if not os.path.exists(self.results_folder):
            print('creating', self.results_folder, 'folder')
            os.makedirs(self.results_folder)

        #-plot overall inertia scores
        plt.figure()
        plt.plot(k_val, IN, linewidth=2, color='y')
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Elbow Method For Optimal k (Inertia)')
        plt.xticks(range(2,highest_k+1))
        plt.savefig( self.results_folder+'/elbow_inertia.png' )

        #-plot overall distortion scores
        plt.figure()
        plt.plot(k_val, DIS, linewidth=2, color='r')
        plt.xlabel('k')
        plt.ylabel('distortion')
        plt.title('Elbow Method For Optimal k (Distortion)')
        plt.xticks(range(2,highest_k+1))
        plt.savefig( self.results_folder+'/elbow_distortion.png' )

        plt.cla()
        plt.clf()
