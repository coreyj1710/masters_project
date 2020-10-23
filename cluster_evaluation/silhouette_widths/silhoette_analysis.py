###BASED ON CODE FROM https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import sklearn.cluster as cluster
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_silouettes(gene_df, clusterer, results_folder):

    print('\nPerforming Silhouette Analysis...')
    n_clusters = len(list( dict.fromkeys(clusterer.labels_) ))

    # Create a subplot with 1 row and 1 column
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(gene_df) + (n_clusters + 1) * 10])

    cluster_labels = clusterer.labels_
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(gene_df, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(gene_df, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        cluster_avg_silhouette = round(sum(ith_cluster_silhouette_values/size_cluster_i),2)

        # Label the silhouette plots with their cluster numbers at the middle
        label = str(size_cluster_i) + ' | ' + str(cluster_avg_silhouette)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, label)

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    #ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("Silhouette coefficient values")

    #heading for text
    ax1.text(-0.052, y_lower + 0.25 * size_cluster_i, 'n = | AVG Width')

    # Clear the yaxis labels
    ax1.set_yticks([])

    #Mark average silhouette
    ##The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red")
    
    ##add label onto x-axis for avg
    plt.xticks([silhouette_avg] + list(plt.xticks()[0]))
    ax1.get_xticklabels()[0].set_color("red")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    plt.suptitle(("Silhouette analysis with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    #create directory to save results
    print('saving silhoette widths results to', results_folder)
    if not os.path.exists(results_folder):
        print('creating \'results_folder\' folder')
        os.makedirs(results_folder)
    plt.savefig( results_folder+'/siloette_widths.png' )
