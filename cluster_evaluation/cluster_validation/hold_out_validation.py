from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.cluster import adjusted_rand_score
import sklearn.cluster as cluster
import statistics
from sklearn.neighbors import NearestCentroid

class HoldOutValidation:

    def __init__(self, gene_df, n_clusters):
        self.gene_df = gene_df
        self.n_clusters = n_clusters

    def kmeans_validation(self, iterations):

        print('\nPerforming holdout validation for kmeans clusterer...')

        rands = []

        for i in range(0,iterations):

            X_train, X_test = train_test_split(self.gene_df, test_size=0.2, random_state=i)

            kmeans_training = KMeans(n_clusters=self.n_clusters, random_state=i).fit(X_train)
            kmeans_testing = KMeans(n_clusters=self.n_clusters, random_state=i).fit(X_test)

            #classify testing data using centroids of training data clusters
            clf = NearestCentroid()
            clf.fit(X_train, kmeans_training.labels_)

            #calculate rand score between clustering labels and prediction labels of held out samples
            rands.append(adjusted_rand_score(clf.predict(X_test), kmeans_testing.labels_))

        ##print('rand scores of kmeans and held out kmeans cluster samples', rands)
        print('average of rand scores', sum(rands) / len(rands))
        print('variance of rand scores', statistics.variance((rands)))

    
    def agglomerative_validation(self, iterations):
        print('\nPerforming holdout validation for agglomerative clusterer...')

        rands = []

        for i in range(0,iterations):

            X_train, X_test = train_test_split(self.gene_df, test_size=0.2, random_state=i)
        
            agglomerative_training = cluster.AgglomerativeClustering( \
                n_clusters=self.n_clusters, linkage='ward', affinity='euclidean').fit(X_train)

            agglomerative_testing = cluster.AgglomerativeClustering( \
                n_clusters=self.n_clusters, linkage='ward', affinity='euclidean').fit(X_test)

            #classify testing data using centroids of training data clusters
            clf = NearestCentroid()
            clf.fit(X_train, agglomerative_training.labels_)

            #calculate rand score between clustering labels and prediction labels of held out samples
            rands.append(adjusted_rand_score(clf.predict(X_test), agglomerative_testing.labels_))
        
        ##print('rand scores of kmeans and held out kmeans cluster samples', rands)
        print('average of rand scores', sum(rands) / len(rands))
        print('variance of rand scores', statistics.variance((rands)))
