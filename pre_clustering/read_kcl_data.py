import numpy as np
import rpy2.robjects as robjects
import pandas as pd

def read_data(root):


    # load your file
    robjects.r['load'](root+'TNBC_data/KCL_TNBC131.RData')

    # retrieve the matrix that was loaded from the file
    matrix = robjects.r['KCL']
    rownames = matrix.rownames
    colnames = matrix.colnames

    # turn the R matrix into a numpy array
    gene_array = np.array(matrix)

    #convert numpy array to pandas df
    gene_df = pd.DataFrame(gene_array)
    gene_df.columns = colnames
    gene_df.index = rownames
    gene_df = gene_df.T

    #label NA feature labels
    features = list(gene_df.columns)
    nul_label = features[len(features)-1]
    for i in range(0, len(features)):
        if features[i] == nul_label:
            features[i] = 'NA_'+str(i)
    gene_df.columns = features

    return gene_df
