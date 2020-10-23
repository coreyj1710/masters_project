from find_k import KFinder
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
#gene_df = cleaner.replace_sparsity_by_median(gene_df)
#gene_df = cleaner.replace_sparsity_by_mean(gene_df)
gene_df = cleaner.remove_sparsity(gene_df)

k_finder = KFinder()
#k_finder.find_k(gene_df, 'results/tcga_find_k_results/', 100, 12)
k_finder.find_k(gene_df, 'results/tcga_find_k_results_cleaned(removal)/', 100, 12)
