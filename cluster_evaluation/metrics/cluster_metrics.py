import sklearn.metrics as metrics
import numpy as np
from sklearn.neighbors import NearestCentroid

class ClusterMetrics:

    def __init__(self, gene_df, clusterer):
        self.gene_df = gene_df
        self.clusterer = clusterer

    '''Metric calculations were inspired by code wriiten during Practical 4: Clustering, as part of the 7CCSMDM1 Data Mining Course at KCL
    https://keats.kcl.ac.uk/pluginfile.php/4706923/mod_resource/content/0/prac4.pdf'''
    def caculate_metrics(self):
        #-compute silhouette score
        SC = metrics.silhouette_score( self.gene_df, self.clusterer.labels_, metric='euclidean' )

        #-compute calinski-harabaz score
        CH = metrics.calinski_harabasz_score( self.gene_df, self.clusterer.labels_ )

        #covert dataframe to numpy array
        genes = self.gene_df.to_numpy()
        
        #get number of clusters
        K = len(list( dict.fromkeys(self.clusterer.labels_) ))

        #-tally members of each cluster
        members = [[] for i in range( K )] # lists of members of each cluster
        for j in range( len( self.gene_df ) ): # loop through instances
            members[ self.clusterer.labels_[j] ].append( j ) # add this instance to cluster returned by scikit function

        #calculate centroids
        nc = NearestCentroid()
        nc.fit(genes, self.clusterer.labels_)

        #-compute the within-cluster score
        within = np.zeros(( K ))
        for i in range( K ): # loop through all clusters
            within[i] = 0.0
            for j in members[i]: # loop through members of this cluster
                # tally the distance to this cluster centre from each of its members
                within[i] += ( np.square( genes[j,0]-nc.centroids_[i][0] ) \
                               + np.square( genes[j,1]-nc.centroids_[i][1] ))
        WC = np.sum( within )

        #-compute the between-cluster score
        between = np.zeros(( K ))
        for i in range( K ): # loop through all clusters
            between[i] = 0.0
            for l in range( i+1, K ): # loop through remaining clusters
                # tally the distance from this cluster centre to the centres of the remaining clusters
                between[i] += ( np.square( nc.centroids_[i][0]-nc.centroids_[l][0] ) \
                                + np.square( nc.centroids_[i][1]-nc.centroids_[l][1] ))
        BC = np.sum( between )

        #-compute overall clustering score
        score = BC / WC

        #-print results for this value of K
        print('\nCluster metrics:')
        print('K = %d,  Within Cluster Score = %.4f,  Between Cluster score = %.4f,  Overall Cluster Score = %.4f, Silhouette = %f,  Calinski-Harabasz = %.4f' \
              % ( K, WC, BC, score, SC, CH ))

    def calculate_variances(self):
        #get list of labels without duplicates
        labels = list(dict.fromkeys(self.clusterer.labels_))
        labels.sort()

        #create copy as to not distort data for further tests
        gene_df = self.gene_df.copy()

        gene_df['label'] = self.clusterer.labels_
        cluster_vars = []
        for label in labels:
            cluster = gene_df.loc[gene_df['label'] == label]
            #covert dataframe to numpy array
            cluster = cluster.to_numpy()
            cluster_vars.append(np.var(cluster))

        
        print('\nCluster Variances: ', cluster_vars)
        print('Average Cluster Variance: ', sum(cluster_vars)/len(cluster_vars))
        
