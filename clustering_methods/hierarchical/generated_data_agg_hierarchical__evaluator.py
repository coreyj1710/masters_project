import sklearn.cluster as cluster
from agglomerative_evaluator import AgglomerativeEvaluator
from sklearn.datasets import make_blobs
import pandas as pd
import sys

#path to root folder
root = '../../'

sys.path.insert(0, root+'pre_clustering/')
import read_tcga_data

sys.path.insert(0, root+'cleaning/')
from data_cleaning import DataCleaning

print('\nPerforming clustering on generated data for comparison...')

#Load data
gene_df = read_tcga_data.read_data(root)
print(gene_df.info())

#clean data
cleaner = DataCleaning()
cleaner.check_sparsity(gene_df)
gene_df = cleaner.remove_sparsity(gene_df)

#get avg std of tcga data features
STD = sum( gene_df.std()) / len(gene_df.std() )
print('average std of features is:', STD)

#generate data with one cluster
X, y = make_blobs(n_samples=112, centers=1, n_features=14085, cluster_std=STD, random_state=0)
generated_df = pd.DataFrame(X)

print(generated_df.info())

results_folder = 'results/generated_agg_results/'

#evaluate clusters
evaluator = AgglomerativeEvaluator(generated_df, results_folder)
evaluator.run_evaluation(100, 3)

