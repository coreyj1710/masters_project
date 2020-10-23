import time
import os
import hashlib
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

'''This class has been left in but was not used in final analysis
due to not producing useful results with the data'''

'''code taken and altered from https://anaconda.org/milesgranger/gap-statistic/notebook'''
class GapStatistic:

    def __init__(self, results_folder):
        self.results_folder = results_folder

    def optimal_k(self, data, nrefs, maxClusters):
        """
        Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
        Params:
            data: ndarry of shape (n_samples, n_features)
            nrefs: number of sample reference datasets to create
            maxClusters: Maximum number of clusters to test for
        Returns: (gaps, optimalK)
        """
        print('\nPerforming gap statistic analysis...')
        gaps = np.zeros((len(range(1, maxClusters)),))
        resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
        for gap_index, k in enumerate(range(1, maxClusters)):

            # Holder for reference dispersion results
            refDisps = np.zeros(nrefs)

            # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
            for i in range(nrefs):
                
                # Create new random reference set
                randomReference = np.random.random_sample(size=data.shape)
                
                # Fit to it
                km = KMeans(k)
                km.fit(randomReference)
                
                refDisp = km.inertia_
                refDisps[i] = refDisp

            # Fit cluster to original data and create dispersion
            km = KMeans(k)
            km.fit(data)
            
            origDisp = km.inertia_

            # Calculate gap statistic
            gap = np.log(np.mean(refDisps)) - np.log(origDisp)

            # Assign this loop's gap statistic to gaps
            gaps[gap_index] = gap
            
            resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

        k = gaps.argmax() + 1  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
        gapdf = resultsdf

        print('Optimal k is: ', k)

        plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
        plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
        plt.grid(True)
        plt.xlabel('Cluster Count')
        plt.ylabel('Gap Value')
        plt.title('Gap Values by Cluster Count')


        #create directory to save results
        if not os.path.exists(self.results_folder):
            print('creating', self.results_folder, 'folder')
            os.makedirs(self.results_folder)
        
        plt.savefig( self.results_folder+'/gap_stat.png' )

        plt.cla()
        plt.clf()
