from find_k import KFinder
import sys

#path to root folder
root = '../'

sys.path.insert(0, root+'pre_clustering/')
import read_kcl_data

#Load data
gene_df = read_kcl_data.read_data(root)
print(gene_df.info())

k_finder = KFinder()
k_finder.find_k(gene_df, 'results/kcl_find_k_results/', 100, 12)
