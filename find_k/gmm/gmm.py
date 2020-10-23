import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from sklearn import decomposition
import seaborn as sns
sns.set()

import os
from datetime import datetime
import pyreadr
import sklearn.mixture as mixture
import matplotlib.pyplot as plt


'inspired by code found at https://towardsdatascience.com/gaussian-mixture-models-d13a5e915c8e'
class Gmm:

    def find_k(gene_df, results_folder, iterations, max_k):
        print('\nPerforming GMM analysis...')
        
        #perform pca (reduce features to equal number of variables)
        pca = decomposition.PCA(n_components = len(gene_df))
        pca.fit(gene_df)
        pca_X = pca.transform(gene_df)

        #time script
        startTime = datetime.now()
        
        n_components = np.arange(1, max_k+1)
        bic = []
        aic = []

        for i in n_components:
            avg_bic = 0
            avg_aic = 0
            for j in range(0, iterations):
                model = GaussianMixture(i, covariance_type='full', random_state=j).fit(pca_X)
                avg_bic += model.bic(pca_X)
                avg_aic += model.aic(pca_X)

            bic.append(avg_bic/iterations)
            aic.append(avg_aic/iterations)

        print('time elapsed', datetime.now() - startTime)

        plt.plot(n_components, bic, label='BIC')
        plt.plot(n_components, aic, label='AIC')
        plt.legend(loc='best')
        plt.xlabel('K')

        #create directory to save results
        if not os.path.exists(results_folder):
            print('creating', results_folder, 'folder')
            os.makedirs(results_folder)

        plt.savefig( results_folder+'/GMM_analysis.png' )

        plt.cla()
        plt.clf()

