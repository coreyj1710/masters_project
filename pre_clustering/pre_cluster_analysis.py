from sklearn.feature_selection import VarianceThreshold

class PreCluster:

    def highest_variance(self, gene_df, top_n):
        
        selector = VarianceThreshold()
        selector.fit_transform(gene_df)

        variances = selector.variances_
        biggest = sorted(range(len(variances)), key=lambda x: variances[x])[-10:]

        print('\nFeatures with highest variance:')
        for i in reversed(biggest):
            print(str(gene_df.columns[i])+':', variances[i])


    
