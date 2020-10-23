from find_k import KFinder
from sklearn.datasets import make_blobs
import pandas as pd
import sys

#path to root folder
root = '../'

sys.path.insert(0, root+'pre_clustering/')
import read_tcga_data

sys.path.insert(1, root+'cleaning/')
from data_cleaning import DataCleaning

#Load data
gene_df = read_tcga_data.read_data(root)
print(gene_df.info())

#clean data - uncomment desired method
cleaner = DataCleaning()
cleaner.check_sparsity(gene_df)
gene_df = cleaner.remove_sparsity(gene_df)

#get avg std of tcga data features
STD = sum( gene_df.std()) / len(gene_df.std() )

#generate data with one cluster
X, y = make_blobs(n_samples=112, centers=1, n_features=14085, cluster_std=STD, random_state=0)
generated_df = pd.DataFrame(X)

k_finder = KFinder()
k_finder.find_k(generated_df, 'results/generated_find_k_results/', 100, 12)
