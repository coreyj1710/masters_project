import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

class DataCleaning:

    def check_sparsity(self, gene_df):
        #display the number of zero values by feature
        pd.set_option('display.max_rows', len(gene_df))
        print('\npresence of 0.0 values:')
        print(gene_df.T.isin([0]).sum())
        null_cols = gene_df.columns[gene_df.isin([0.0]).any()]
        print('number of columns with 0.0 values:', len(null_cols), 'of', len(gene_df.columns))

    def remove_sparsity(self, gene_df):
        print('\nremoving cols with zero values from data...')
        null_cols = gene_df.columns[gene_df.isin([0.0]).any()]
        gene_df = gene_df.drop(columns=null_cols)
        gene_df = pd.DataFrame(gene_df)
        return gene_df

    def replace_sparsity_by_mean(self, gene_df):
        gene_df = gene_df.replace({0.0:np.nan})
        imputer = SimpleImputer(missing_values=np.nan, strategy='median').fit(gene_df)
        gene_df = imputer.transform(gene_df)
        gene_df = pd.DataFrame(gene_df)
        return gene_df

    def replace_sparsity_by_median(self, gene_df):
        gene_df = gene_df.replace({0.0:np.nan})
        imputer = SimpleImputer(missing_values=np.nan, strategy='median').fit(gene_df)
        gene_df = imputer.transform(gene_df)
        gene_df = pd.DataFrame(gene_df)
        return gene_df
