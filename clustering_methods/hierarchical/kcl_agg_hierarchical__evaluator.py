from agglomerative_evaluator import AgglomerativeEvaluator
import sys

#path to root folder
root = '../../'

sys.path.insert(0, root+'pre_clustering/')
import read_kcl_data

#Load data
gene_df = read_kcl_data.read_data(root)
print(gene_df.info())

print('\nPerforming agglomerative clustering on kcl dataset...')

results_folder = 'results/kcl_agg_results/'

#evaluate clusters
evaluator = AgglomerativeEvaluator(gene_df, results_folder)
evaluator.run_evaluation(100, 4)
