import sklearn.cluster as cluster
from agglomerative_evaluator import AgglomerativeEvaluator
import sys

#path to root folder
root = '../../'

sys.path.insert(0, root+'pre_clustering/')
import read_tcga_data

sys.path.insert(0, root+'cleaning/')
from data_cleaning import DataCleaning

#Load data
gene_df = read_tcga_data.read_data(root)
print(gene_df.info())

#clean data
cleaner = DataCleaning()
cleaner.check_sparsity(gene_df)
gene_df = cleaner.remove_sparsity(gene_df)
##gene_df = cleaner.replace_by_mean(gene_df)
##gene_df = cleaner.replace_by_median(gene_df)

print('\nPerforming agglomerative clustering on tcga dataset...')

results_folder = 'results/tcga_agg_results_cleaned(removal)/'

#evaluate clusters
evaluator = AgglomerativeEvaluator(gene_df, results_folder)
evaluator.run_evaluation(100, 3)
