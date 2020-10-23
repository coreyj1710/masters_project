import pyreadr

def read_data(root):

    #read dataset
    result = pyreadr.read_r(root+'TNBC_data/TCGA_TNBC112.RData')

    print(result.keys()) # let's check what objects we got
    gene_df = result["TCGA"] # extract the pandas data frame for object df1

    return gene_df.T

