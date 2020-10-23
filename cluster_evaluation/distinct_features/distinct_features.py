import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

class DistinctFeatures:

    def __init__(self, gene_df, results_folder):
        self.gene_df = gene_df
        self.results_folder = results_folder
        print('\nFinding most distinct features between clusters...') 

    def find_features(self, clusterer, n_features):
        gene_df = self.gene_df
        results_folder = self.results_folder

        #get list of labels without duplicates
        labels = list(dict.fromkeys(clusterer.labels_))
        labels.sort()

        #create copy as to not distort data for further tests
        gene_df = self.gene_df.copy()

        #separte clusters 
        features = gene_df.columns
        gene_df['label'] = clusterer.labels_
        feature_avgs = []
        for label in labels:
            #get feature averages for each cluster
            cluster = gene_df.loc[gene_df['label'] == label]
            cluster = cluster.drop(['label'], axis=1)
            feature_avg = cluster.mean().to_numpy()
            feature_avgs.append(feature_avg)

        #Convert feature averages to a dataframe
        cluster_df = pd.DataFrame(data=feature_avgs, columns=features)

        #Remove 1st cluster (Added to further analyse KCL dataset
        ##cluster_df = cluster_df.drop([0])
        
        #find features with most variance
        selector = VarianceThreshold()
        selector.fit_transform(cluster_df)
        variances = selector.variances_
        biggest = sorted(range(len(variances)), key=lambda x: variances[x])[-n_features:]
        print('\nFeatures with highest variance:')
        variant_features = []
        for i in reversed(biggest):
            variant_features.append(cluster_df.columns[i])
            print(str(cluster_df.columns[i])+':', variances[i])

        print()
        print(cluster_df[variant_features].to_string())

