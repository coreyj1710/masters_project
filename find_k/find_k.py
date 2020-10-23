import sys

sys.path.insert(0, 'elbow/')
from elbow import Elbow

sys.path.insert(0, 'elbow/')
from elbow import Elbow

sys.path.insert(0, 'consensus_cluster/')
from consensus_clustering import ConsensusClustering

sys.path.insert(0, 'gmm/')
from gmm import Gmm

sys.path.insert(0, 'gap_statistic/')
from gap_statistic import GapStatistic


class KFinder():

    def find_k(self, gene_df, results_folder, iterations, max_k):
        print('\nRunning evaluation to find optimal K...')

        #perform elbow method
        elbow = Elbow(results_folder+'elbow/')
        elbow.elbow_method(gene_df, iterations, max_k)

        #perform consensus clustering
        consensus = ConsensusClustering(results_folder+'consensus/')
        consensus.combined_consensus(gene_df, iterations, 2, 5)
        consensus.consensus_clustering_evaluation(gene_df, iterations, max_k)

        #perform gmm analysis for optimal k
        '''Gmm can fail when run with consensus clustering. For this reason it is recommended to not run both at once'''
        ##Gmm.find_k(gene_df, results_folder+'gmm/', iterations, max_k)

        #Perform gap statistic analysis
        gap_statistic = GapStatistic(results_folder+'gap_stat/')
        gap_statistic.optimal_k(gene_df, iterations, max_k)

