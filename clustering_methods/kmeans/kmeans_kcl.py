from kmeans_evaluator import KMeansEvaluator
import sys

#path to root folder
root = '../../'

sys.path.insert(0, root+'pre_clustering/')
import read_kcl_data
from pre_cluster_analysis import PreCluster

sys.path.insert(0, root+'cleaning/')
from data_cleaning import DataCleaning

#Load data
gene_df = read_kcl_data.read_data(root)
print(gene_df.info())

#find highest variance features
pre_cluster = PreCluster()
pre_cluster.highest_variance(gene_df, 10)

#clean data
cleaner = DataCleaning()
cleaner.check_sparsity(gene_df)

print('\nPerforming evaluation of k-means clustering on kcl dataset...')

#specify folder name to save results to
results_folder = 'results/kcl_kmeans_results'

#initialise kmeans evaluator
evaluator = KMeansEvaluator(gene_df, results_folder)

#evaluate kmeans clusters
evaluator.run_evaluation(100, 4)

