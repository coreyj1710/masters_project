import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import os
import sklearn.cluster as cluster


class DendrogramPlotter:

    def __init__(self, gene_df, results_folder):
        self.gene_df = gene_df
        self.results_folder = results_folder

        #print('saving results...')
        if not os.path.exists(self.results_folder):
            print('creating', self.results_folder, 'folder')
            os.makedirs(self.results_folder)
        
    #plot sub dendrograms of found clusters
    def plot_sub_dendrograms(self, n_clusters):
        print('Creating sub sub-dendrograms for each cluster...')
        ac = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', affinity='euclidean')
        ac.fit(self.gene_df)
        self.gene_df['label'] = ac.labels_

        for i in range(0, n_clusters):
            plt.figure(figsize=(10, 7))
            sub_cluster_df = self.gene_df.loc[self.gene_df['label'] == i]
            sub_cluster_df.drop(columns=['label'])
            sub_dend = shc.dendrogram(shc.linkage(sub_cluster_df, method='ward'))
            plt.savefig( self.results_folder+'sub_dendrograms_cluster_'+str(i+1)+'.png' )

            plt.cla()
            plt.clf()
            plt.close()

    def plot_dendrogram(self):
        print('Creating dendrogram from data...')
        #plot dendogram
        plt.figure(figsize=(10, 7))
        dend = shc.dendrogram(shc.linkage(self.gene_df, method='ward'))
        plt.savefig( self.results_folder+'/dendrogram.png' )
        
        plt.cla()
        plt.clf()
        plt.close()
