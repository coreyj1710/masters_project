I verify that I am the sole author of the programmes contained in this archive, except where explicitly stated to the contrary.
- Corey jenkins


This is the supplementary file for my project which analyses TNBC gene expression data. In this text document I will explain how to run 
the code in the file. There are several ways to run the different tests implemented. I will explain which files are used for which tests. 
The following files can be run by opening them in an environment such as IDLE then running them. The code was written and run in Python 3.7.
It is recommended to use Python 3.7 when running it.


Finding the number of clusters:
- To run, navigate to the find_k folder. Inside this folder are several python files. Each one will run the tests to find k for a given dataset:
	- find_k_kcl.py for kcl dataset
	- find_k_tcga.py for tcga dataset
	- find_k_generated.py for randomly generated dataset
- The tests to execute are specified inside find_k.py which is called by the previously mentioned Python files.
- Some tests are commented out in find_k.py. This will be explained further at the end of this document.


Clustering and evaluation:
- To perform clustering methods and evaluate the clusters, navigate to clustering_methods folder. Inside there are two folders called hierarchical
  and kmeans which contain functionality for each method.

- kmeans:
	- kmeans folder contains several files for k-means clustering, running each file clusters the given dataset:
		- kmeans_kcl.py for kcl dataset
		- kmeans_tcga.py for tcga dataset
	- The separate evaluation methods to execute are specified inside kmeans_evaluator.py which is called in the previously mentioned files.

- hierarchical:
	- hierarchical folder contains several files for hierarchical clustering, running each file fclusters the given data:
		- kcl_agg_hierarchical__evaluator.py for kcl dataset
		- tcga_agg_hierarchical__evaluator.py for tcga dataset
		- generated_data_agg_hierarchical__evaluator.py for randomly generated dataset
	- The separate evaluation methods to execute are specified inside agglomerative_evaluator.py which is called in the previously mentioned files.
	- Some tests are commented out in agglomerative_evaluator.py. This will be explained further at the end of this document.


Each of the folders mentioned have a results folder in which results of the evaluations will be saved to.

WARNING!!!
Certain tests and evaluations were found to fail when run alongside each other. This can happen when consensus clustering is run alongside GMM analysis
(these are both called in find_k.py), and when dendrogram plotting runs alongside pca analysis (these are both called in agglomerative_evaluator.py).
For this reason, the execution of some of these tests have been commented out. To run them, simply swap which tests are commented out.
In researching the issue it is thought to possibly be caused by a bug with sklearn when performing PCA. PCA is used in both pca analyis (obviously) and in the
implementation of GMM analysis.
