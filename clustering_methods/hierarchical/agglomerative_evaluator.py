import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn import decomposition
import numpy as np
from dendrogram_plotter import DendrogramPlotter

import sys

#path to root folder
root = '../../'

sys.path.insert(0, root+'cluster_evaluation/metrics/')
from cluster_metrics import ClusterMetrics

sys.path.insert(0, root+'cluster_evaluation/pca/')
from pca_plotter import PcaPlotter

sys.path.insert(0, root+'cluster_evaluation/cluster_validation/')
from hold_out_validation import HoldOutValidation

sys.path.insert(0, root+'cluster_evaluation/silhouette_widths/')
import silhoette_analysis

sys.path.insert(0, root+'cluster_evaluation/distinct_features/')
from distinct_features import DistinctFeatures


class AgglomerativeEvaluator:

    def __init__(self, gene_df, results_folder):
        self.gene_df = gene_df
        self.results_folder = results_folder  

    def run_evaluation(self, iterations, k):

        gene_df = self.gene_df
        results_folder = self.results_folder

        #create dendrograms
        '''ploting dendrograms can cause pca plotting to fail. For this reason it is recommended to not run both at once'''
        ##dendro_plotter = DendrogramPlotter(gene_df.copy(), results_folder+'dendrograms/')
        ##dendro_plotter.plot_dendrogram()
        ##dendro_plotter.plot_sub_dendrograms(k)

        #create clusters using sckit-learn's agglomerative clustering function
        ac = cluster.AgglomerativeClustering(n_clusters=k, linkage='ward', \
                                             affinity='euclidean').fit(gene_df)

        #compute cluster metrics
        gene_df = self.gene_df.copy()
        c_metrics = ClusterMetrics(gene_df, ac)
        c_metrics.caculate_metrics()
        c_metrics.calculate_variances()

        #holdout validation
        n_clusters = len(list( dict.fromkeys(ac.labels_) ))
        validator = HoldOutValidation(gene_df, n_clusters)
        validator.agglomerative_validation(iterations)

        #plot sillhoette widths
        silhoette_analysis.plot_silouettes(gene_df, ac, results_folder+'silhouette_analysis/')

        #Find distinct features in clusters
        distinct = DistinctFeatures(gene_df, results_folder+'distinct_features/')
        distinct.find_features(ac, 10)

        #Use pca to visualise clusters
        pca_plotter = PcaPlotter(results_folder+'pca/')
        pca_plotter.find_most_important(gene_df, ac, 10)
        pca_plotter.plot_2d(gene_df, ac)
        pca_plotter.plot_3d(gene_df, ac)
        print('\nFinished cluster evaluation')
