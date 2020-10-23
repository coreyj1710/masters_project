from kmeans_evaluator import KMeansEvaluator
from sklearn.cluster import KMeans

#path to root folder
root = '../../'

import sys
sys.path.insert(0, root+'pre_clustering/')
import read_tcga_data
from pre_cluster_analysis import PreCluster

sys.path.insert(0, root+'cleaning/')
from data_cleaning import DataCleaning

#Load data
gene_df = read_tcga_data.read_data(root)
print(gene_df.info())

#find highest variance features
pre_cluster = PreCluster()
pre_cluster.highest_variance(gene_df, 10)

#clean data
cleaner = DataCleaning()
cleaner.check_sparsity(gene_df)
gene_df = cleaner.remove_sparsity(gene_df)
##gene_df = cleaner.replace_by_mean(gene_df)
##gene_df = cleaner.replace_by_median(gene_df)

#find highest variance features
pre_cluster = PreCluster()
pre_cluster.highest_variance(gene_df, 10)

print('\nAnalysing kmeans clustering on tcga dataset...')

#specify folder name to save results to
results_folder = 'results/tcga_kmeans_results_cleaned(removal)/'

#initialise kmeans evaluator
evaluator = KMeansEvaluator(gene_df, results_folder)

#evaluate kmeans clusters
evaluator.run_evaluation(100, 5)
