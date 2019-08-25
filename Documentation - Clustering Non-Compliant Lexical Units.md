# Clustering Non-Compliant Lexical Units

This documentation details the process of clustering the lexical units which are non-existent in Berkeley FrameNet 1.7 and which do not pass through either the POS or Core FEs valence pattern checking.

All the necessary files reside in the folder `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs` and the file that is run is `cluster_LUs_AP.py`.

The `sbatch` script used to generate the t-SNE visualization and the clustering of the lexical units which are filtered out by the POS and CoreFEs filters:

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=7-00:00:00
#SBATCH --output=clutser_LUs_AP_output.stdout
#SBATCH --error=clutser_LUs_AP_output.err
#SBATCH --job-name="clustering-unmatched-LUs"

module load gcc/6.3.0 openmpi/2.0.1 python/3.6.6
module load singularity
export SINGULARITY_BINDPATH="/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs:/mnt"

singularity exec production.sif python3 -u /mnt/cluster_LUs_AP.py --folder_unmatched_lus="/mnt/data/0102" --folder_models="/mnt/data/0102" --folder_images="/mnt/data" > './data/0102/clutser_LUs_AP_output.out'
```

The argument `--folder_unmatched_lus` specifies the folder that currently stores the unmatched LUs pickled file whose file names starts with either "unmatched_coreFEs_lus" or "unmatched_pos_lus" (such as "unmatched_pos_lus-2019-01-02_1600_DE_DasErste_Tagesschau.seg.p"). Remember that it must use SINGULARITY_BINDPATH.

The argument `--folder_models` specifies the folder to store the file (`lu_cluster_affinity_propagation.pkl`). Remember that it must use SINGULARITY_BINDPATH.

The argument `--folder_images` specifies the folder to store the visualization image of the clusters. Remember that it must use SINGULARITY_BINDPATH.

The final outputs are:

- a tuple of (unmatched_LUs_to_tensors, X, LUs, cluster_centers_indices, labels): `{folder_models}/lu_cluster_affinity_propagation.pkl`
  - *unmatched_LUs_to_tensors*: a dictionary of unseen filtered-out LUs (whose POS does not exist in the frame of its closest lexical units OR whose exemplar sentences contain core FEs in the frame of its closest lexical units) mapped to their BERT embeddings representations
  - *X*: unmatched_LUs_to_tensors.values(), which are a list of BERT embeddings representations in the dictionary of unseen filtered-out LUs
  - *LUs*: unmatched_LUs_to_tensors.keys(), which are a list of unseen filtered-out LUs
  - *cluster_centers_indices*: indices of cluster centers
  - *labels*: cluster label of each LUs

- t-SNE Visualization of lexical units: `{folder_images}/viz_LU_vectors.png`
- t-SNE Visualization of lexical units which are clustered: `{folder_images}/viz_clustered_LUs.png`

---

## Implementation Details

1. **t-SNE**

t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear technique for dimensionality reduction that helps visualize high-dimensional datasets. It maps the multi-dimensional tensor (embedding of LUs) to a lower dimensional space.

In my implementation, `X_embedded` is the lower-dimensional embeddings that is mapped from `X` (a list of BERT embeddings representations in the dictionary of unseen filtered-out LUs). 

If `cluster_centers_indices` is given, which is the centers of the clusters, the data points in the 3D space which are in the same clusters (known from `labels`, which is obtained from the `sklearn.cluster.AffinityPropagation` clustering model).

```python
def visualize(X, LUs, save_fig_name, cluster_centers_indices=None, labels=None):
    X_embedded = TSNE(n_components=3, perplexity=40, n_iter=2500, random_state=23).fit_transform(X)
    plt.close('all')
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)

    if cluster_centers_indices is None and labels is None:
        for i, x in enumerate(X_embedded):
            ax.scatter(x[0], x[1], x[2])

    elif cluster_centers_indices is not None and labels is not None:
        colors = itertools.cycle('bgrcmyk')
        no_clusters = len(cluster_centers_indices)
        for k, color in zip(range(no_clusters), colors):
            cluster_members = labels == k
            cluster_center = X_embedded[cluster_centers_indices[k]]
            ax.plot(X_embedded[cluster_members, 0], X_embedded[cluster_members, 1], X_embedded[cluster_members, 2], color + '.')
            ax.plot([cluster_center[0]], [cluster_center[1]], [cluster_center[2]], 'o',
                    markeredgecolor='k', markersize=5, markerfacecolor=color)
            for i, x in enumerate(X_embedded[cluster_members]):
                ax.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], [cluster_center[2], x[2]], color)

    elif cluster_centers_indices is None:
        raise AssertionError("Empty cluster_centers_indices")
    elif labels is None:
        raise AssertionError("Empty labels")

    # annotate the LUs vectors with the LU name
    for i, x in enumerate(X_embedded):
        ax.text(x[0], x[1], x[2], '%s' % (LUs[i]), size=5, zorder=1)


    plt.savefig(save_fig_name)
    plt.show()
```



**Visualizations**

The following figure shows the LUs' embeddings represented in a 3D space.

![viz_LU_vectors](https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/images/viz_LU_vectors.png)

The following figure shows the clustered LUs' embeddings represented in a 3D space.

![viz_clustered_LUs](https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/images/viz_clustered_LUs.png)

2. **Affinity Propagation**

Affinity Propagation is a clustering algorithm that does not require the number of clusters to be determined beforehand. It is therefore suitable for the task of clustering unseen lexical units as we do not know how many frames (clusters) would these new lexical units group into. 

```python
def affinity_propagation_cluster(X):
    af = AffinityPropagation(damping=0.5, max_iter=500, affinity='euclidean').fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    return cluster_centers_indices, labels
```

**Analysis of Clustering Results**

From the above figure of clustered LUs' embeddings, we notice that if we use the default preference parameter of `AffinityPropagation`, which is the median of the input similarities, the clustering is not accurate despite promising.

For example, the green cluster ("legislation.n", "council.n", "councilman.n") is correctly clustered but the rest of the clusters seem to have a lot of false positives. For example, the yellow cluster ("refill.n", "refund.n", "prescription.n") should not include "prescription.n". 

A potential reason for this inaccuracy is that the exemplar sentences are too few to shed semantic differences between the three lexical units. All of them could fit in the sentence "A customer ask for [refill / refund / prescription].", which seems to suggest that all three of them could be in the same frame.

It is important to note that for now, I could not find the best `preference` hyperparameter value because currently, I could not find any way to evaluate to performance of the clustering aside from analyzing the visual model. 


---
## Documentation of Singularity Containers
I reuse the Singularity container `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/production.sif` that is used for the task of [Expanding FrameNet with NewsScape and Embedding](https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/Documentation%20-%20Expanding%20FrameNet%20with%20NewsScape%20and%20Embedding.md#documentation-of-creating-singularity-containers)

The Python 3 external libraries/dependencies imported in the file `cluster_LUs_AP.py` are `sklearn`, `numpy`, `torch`, and `matplotlib`.
