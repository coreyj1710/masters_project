import pyreadr
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as robjects
import pandas as pd
import os

class PcaPlotter:

    def __init__(self, results_folder):
        print('\nReducing dimensionality for visualisation...')
        
        #create directory to save results
        self.results_folder = results_folder
        if not os.path.exists(self.results_folder):
            print('\ncreating \'results_folder\' folder')
            os.makedirs(self.results_folder)

    #reduce data to 3d and visualise
    def plot_3d(self, gene_df, clusterer):
        print('Reducing data clusters to 3D')
        pca = decomposition.PCA(n_components = 3)
        pca.fit(gene_df)
        pca_X = pca.transform(gene_df)

        #calculate information loss
        information = sum(pca.explained_variance_ratio_)
        print('Recovered variance after PCA:',information)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colormap = np.array(['r', 'g', 'b','y','c'])
        #colormap = np.array([1,2,3,4,5,6,7])
        ax.scatter(pca_X[:,0], pca_X[:,1],pca_X[:,2], c=colormap[clusterer.labels_]);
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        plt.show()
        plt.close()

    #reduce data to 2d and visualise
    def plot_2d(self, gene_df, clusterer):
        print('Reducing data clusters to 2D')

        pca = decomposition.PCA(n_components = 2)
        pca.fit(gene_df)
        pca_X = pca.transform(gene_df)

        #calculate information loss
        information = sum(pca.explained_variance_ratio_)
        print('Recovered variance after PCA:',information)

        fig = plt.figure()
        ax = fig.add_subplot()

        colormap = np.array(['r', 'g', 'b','y','c'])
        #colormap = np.array([1,2,3,4,5,6,7])
        ax.scatter(pca_X[:,0], pca_X[:,1], c=colormap[clusterer.labels_]);
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        print('Saving 2d plot to', self.results_folder)
        plt.savefig(self.results_folder+'/pca_plot_2d.png' )

    def find_most_important(self, gene_df, clusterer, n_pcs):
        '''code for this function found at https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis'''
        print('Finding most important features for principal components...') 
        pca = decomposition.PCA(n_components = n_pcs).fit(gene_df)
        pca_X = pca.transform(gene_df)

        # get the index of the most important feature on EACH component
        # LIST COMPREHENSION HERE
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

        initial_feature_names = list(gene_df.columns)
        # get the names
        most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

        # LIST COMPREHENSION HERE AGAIN
        dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

        # build the dataframe
        df = pd.DataFrame(dic.items())

        print(df.to_string())
