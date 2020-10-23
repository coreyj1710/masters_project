import sklearn.metrics as metrics
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from sklearn.cluster import KMeans

#path to root folder
root = '../../'

from sklearn.cluster import KMeans
import sys

sys.path.insert(0, root+'cluster_evaluation/cluster_validation/')
from hold_out_validation import HoldOutValidation

sys.path.insert(0, root+'cluster_evaluation/pca')
from pca_plotter import PcaPlotter

sys.path.insert(0, root+'cluster_evaluation/silhouette_widths/')
import silhoette_analysis

sys.path.insert(0, root+'cluster_evaluation/metrics/')
from cluster_metrics import ClusterMetrics


class KMeansEvaluator:
    '''This class was created to evaluate metrics for different values of k.
    The metrics are averaged over multiple iterations for multiple k values.
    This can be used to help decide an otimal number for k.'''

    def __init__(self, gene_df, results_folder):
        self.gene_df = gene_df
        self.results_folder = results_folder

    def run_evaluation(self, iterations, k):
        
        gene_df = self.gene_df
        results_folder = self.results_folder

        #perfomr clustering
        kmeans = KMeans(n_clusters=k, random_state=0).fit(gene_df)

        #compute cluster metrics
        c_metrics = ClusterMetrics(gene_df, kmeans)
        c_metrics.caculate_metrics()
        c_metrics.calculate_variances()

        #perform hold out validation
        n_clusters = len(list( dict.fromkeys(kmeans.labels_) ))
        validator = HoldOutValidation(gene_df, n_clusters)
        validator.kmeans_validation(iterations)

        #plot sillhoette widths
        silhoette_analysis.plot_silouettes(gene_df, kmeans, results_folder)

        #Use pca to visualise clusters
        pca_plotter = PcaPlotter(results_folder)
        pca_plotter.plot_2d(gene_df, kmeans)
        pca_plotter.plot_3d(gene_df, kmeans)

        print('\nFinished cluster evaluation')
