from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from scipy.integrate import simps
from numpy import trapz

'''code for producing a consensus matrix was first found used in a study on
kaggle.com called ensemble clustering. This can be found at
https://www.kaggle.com/noise42/ensemble-clustering.
Changes were made such as splitting the analysis to individual k values and
adding further functionality like calculating cumulative distribution function
for finding the optimal k value.'''
def ensemble_kmeans(data, n_random_states, k_list):
    #Run clustering with different k and check the metrics
    labs=[]
    for r in range(0, n_random_states):
        for k in k_list:
            clusterer = KMeans(n_clusters=k, random_state=r)
            clusterer.fit(data)
            labs.append(clusterer.labels_)
    return np.array(labs)

#construct a cooccurrence (consensus) matrix
def cons_matrix(labels):
    C=np.zeros([labels.shape[1],labels.shape[1]], np.int32)
    for label in labels:
        for i, val1 in enumerate(label):
            for j, val2 in enumerate(label):                   
                if val1 == val2 :
                    C[i,j] += 1 
                
    return pd.DataFrame(C)

#get cumulative distribution function
def cdf(consensus_matrix):
    consensus_values = []
    depth = 1
    for i in range(1, consensus_matrix.shape[0]):
        for j in range(0, depth):
            consensus_values.append(consensus_matrix[i,j-1])
        depth += 1
            
    return np.sort(consensus_values)

class ConsensusClustering:

    def __init__(self, results_folder):
        print('\nPerforming consensus clustering...')
        #create directory to save results
        if not os.path.exists(results_folder):
            print('Creating', results_folder, 'folder')
            os.makedirs(results_folder)

        self.results_folder = results_folder

    #iterate over different Ks and plot CDFs
    def consensus_clustering_evaluation(self, gene_df, n_random_states, max_k):
        print('Creating consensus matrices for individual Ks')

        results_folder = self.results_folder

        #time script
        startTime = datetime.now()

        #create directory to save results
        if not os.path.exists(results_folder):
            print('creating', results_folder, 'folder')
            os.makedirs(results_folder)

        #for each value of k calculate its consensus matrix and plot results
        areas_of_k = []
        for k in range(2, max_k+1):
            print('processing k=',k,'out of',max_k)
            klist = [k]
            cl_data=ensemble_kmeans(gene_df, n_random_states, klist)

            consensus_matrix = cons_matrix(cl_data)
            consensus_matrix.columns= gene_df.index
            consensus_matrix.index=gene_df.index

            clustermap=sns.clustermap(consensus_matrix)
            plt.setp(clustermap.ax_heatmap.get_yticklabels(), rotation=0)
            # remove dendrograms from side of plot
            clustermap.ax_row_dendrogram.set_visible(False)
            clustermap.ax_col_dendrogram.set_visible(False)
            # Use the dendrogram box to reposition the colour bar
            dendro_box = clustermap.ax_row_dendrogram.get_position()
            dendro_box.x0 = (dendro_box.x0 + 2 * dendro_box.x1) / 3
            clustermap.cax.set_position(dendro_box)
            # Move the ticks to the left
            clustermap.cax.yaxis.set_ticks_position("left")
            plt.savefig( results_folder+'/k='+str(k)+'.png' )

            consensus_matrix = consensus_matrix.to_numpy()

            x = cdf(consensus_matrix)
            norm_x = [float(i)/n_random_states for i in x]
            
            y = range(1,len(x)+1)
            norm_y = [float(i)/len(y) for i in y]

            areas_of_k.append(trapz(norm_x))

            plt.figure('cdf')
            plt.plot(norm_x, norm_y, label = k)
            plt.xlabel('Consensus index')
            plt.ylabel('Cumulative Distribution Function (CDF)')
            leg = plt.legend();
            if k==max_k:
                plt.savefig( results_folder+'/cdf.png' )

        print('time elapsed:', datetime.now() - startTime)

        plt.figure('areas')

        plt.plot(range(2, max_k+1), areas_of_k, linestyle='--', marker='o', color='b')
        plt.xlabel('k')
        plt.ylabel('area under curve')
        plt.savefig( results_folder+'/cdf_areas.png' )

        plt.cla()
        plt.clf()

    #create one consensus matrix combining multiple Ks
    def combined_consensus(self, gene_df, n_random_states, min_k, max_k):

        #time script
        startTime = datetime.now()

        print('Creating consensus matrix for multiple Ks')
        results_folder = self.results_folder

        k_list = range(min_k,max_k+1)
        cl_data = ensemble_kmeans(gene_df, n_random_states, k_list)

        consensus_matrix = cons_matrix(cl_data)
        consensus_matrix.columns = gene_df.index
        consensus_matrix.index = gene_df.index

        clustermap=sns.clustermap(consensus_matrix)
        plt.setp(clustermap.ax_heatmap.get_yticklabels(), rotation=0)
        # remove dendrograms from side of plot
        clustermap.ax_row_dendrogram.set_visible(False)
        clustermap.ax_col_dendrogram.set_visible(False)
        # Use the dendrogram box to reposition the colour bar
        dendro_box = clustermap.ax_row_dendrogram.get_position()
        dendro_box.x0 = (dendro_box.x0 + 2 * dendro_box.x1) / 3
        clustermap.cax.set_position(dendro_box)
        # Move the ticks to the left
        clustermap.cax.yaxis.set_ticks_position("left")

        print('Time elapsed:', datetime.now() - startTime)
        
        plt.savefig( results_folder+'/consensus_matrix, K='+str(min_k)+' to k='+str(max_k)+'.png' )
