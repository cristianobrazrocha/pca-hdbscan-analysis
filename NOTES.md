# Method notes: shared PCA and HDBSCAN

## Why is this Analysis Useful? 

Trajectories and multivariate measurements contain more information than is easy to read one observation at a time. For Molecular Dynamics, analyzing the conformational space of Intrinsically Disordered Proteins (IDPs) is notoriously difficult because they sample a vast, highly heterogeneous conformational landscape.

This pipeline places several systems on one set of PCA axes, identifies densely sampled regions within each system, and retrieves original observations representing those regions.

For MD trajectories, PCA is performed using the non-mass-weighted covariance matrix of atomic displacements (e.g., Cα atoms). To dissect the conformational ensemble, HDBSCAN clustering is applied to the PCA-projected coordinates. This density-based hierarchical algorithm identifies regions of high local point density separated by low-density boundaries, allowing the detection of clusters with arbitrary shapes without needing to predefine the number of clusters. This makes HDBSCAN particularly suitable for IDPs, whose conformational ensembles are non-Gaussian and often characterized by diffuse, partially overlapping basins.

For general tabular data (like the sensor tutorial), it similarly shows operating modes shared across conditions and isolates newly introduced behaviors. Note that HDBSCAN can leave sparse observations unassigned as noise. These patterns do not by themselves establish a molecular mechanism, physical metastability, equipment failure, or predictive accuracy.

## What the code does

1. **Prepare features.** In MD, select atoms in corresponding order, align
   frames to the reference system's first frame, and flatten their selected
   Cartesian coordinates. For CSV, read the ordered numeric `features` from
   each row; there is no alignment or automatic scaling.
2. **Fit shared PCA.** Pool all systems, center using **one pooled mean**, fit
   one PCA model and project every system with that same mean and basis.
   Systems with more observations contribute more to the fitted basis.
3. **Cluster separately.** Fit the Python package's `hdbscan.HDBSCAN` to
   each system's retained PC coordinates with Euclidean distance. The code
   sets `min_cluster_size = max(5, int(min_cluster_fraction × n_system))`.
   The package defaults to `min_samples = min_cluster_size` and excess-of-mass
   (`eom`) cluster selection. The script relabels clusters by size within
   each system (0 largest); `-1` means noise.
4. **Export observed medoids.** For each cluster, select a sampled point
   minimizing summed distances to all other cluster points in **all retained
   PCs**. Recover its full source trajectory frame as a PDB or its source
   CSV row as a CSV. Figures display only PC1 and PC2.

For trajectories, `reference` selects both the alignment frame and the gray
background in overlay plots. For CSV, it only selects that background. PCA
is fitted to pooled data, **not** to the reference alone; the code never
subtracts the reference observations from the other system.

## Equations behind the outputs

Let $x_i$ be a $D$-feature observation, and $N$ the total count pooled across
systems. With pooled mean $\bar{x}=N^{-1}\sum_{i=1}^{N}x_i$, PCA uses

$$C=\frac{1}{N-1}\sum_{i=1}^{N}(x_i-\bar{x})(x_i-\bar{x})^\mathsf{T}.$$

For covariance eigenvector $v_j$, the score is
$z_{ij}=v_j^\mathsf{T}(x_i-\bar{x})$. The explained variance fraction is
$\lambda_j/\sum_k\lambda_k$, where $\lambda_j$ is the corresponding
eigenvalue. This is unweighted Cartesian PCA: no per-system re-centering,
atomic masses, or feature scaling is applied.

HDBSCAN uses a **core distance** to express local sparseness. Informally,
$\mathrm{core}_k(a)=d(a,N_k(a))$, the distance to the $k$th neighbor
of $a$, where $k$ reflects `min_samples`. In retained PC space the ordinary
distance is $d(a,b)=\lVert z_a-z_b\rVert_2$. The **mutual reachability**
distance is

$$d_{\mathrm{mreach},k}(a,b)=\max\{\mathrm{core}_k(a),
\mathrm{core}_k(b),d(a,b)\}.$$

The algorithm builds a minimum spanning tree from these distances and a
hierarchy of density-connected groups. Small branches are pruned. With
$\lambda=1/d_{\mathrm{mreach}}$, the stability of a candidate cluster can
be described by

$$S(C)=\sum_{p\in C}\bigl(\lambda_p-
\lambda_{\mathrm{birth}}(C)\bigr),$$

where $\lambda_p$ is the level at which $p$ leaves $C$. Default EOM
selection compares a candidate's stability with its descendants' to
extract a set of clusters. This is persistence in a **sampled point cloud**,
not evidence of a physical free-energy basin. The original papers and
package guide below explain the algorithm more fully.

For a cluster $C$ with projected observations $z_i$, its sampled medoid is

$$i^*=\underset{i\in C}{\mathrm{arg\,min}}\;\sum_{j\in C}\lVert z_i-z_j\rVert_2.$$

The table's occupancy for cluster label $c$ is

$$f_c=100\,\frac{n_c}{N}.$$
{\#\{\text{all observations in that system}\}}.$$

Noise remains in the denominator. `min_cluster_fraction` is this script's
way to choose a minimum cluster size; it is not a universal parameter
recommendation.

## Reading the sensor overlay

The gray baseline has two dense modes. The orange changed data overlap
those modes and introduce a third group farther right. The axes are pooled
PC1 and PC2, not time, sensor names, or a fault score. Gray and orange
identify **input conditions**. Separate `pca_scatter.png` files colour the
**HDBSCAN clusters**. An isolated dot or a noise label is not a verified
anomaly. With real measurements, inspect original values, calibration,
sampling, and operating metadata before drawing a causal conclusion.

## Limits worth checking

- The 2D plot omits PC3 onward; HDBSCAN uses every retained PC. Numeric PC
  values cannot be compared between separately fitted analyses.
- Unequal system sizes weight the pooled PCA. Correlated adjacent frames or
  sensor readings, sampling imbalance and HDBSCAN parameters can all change
  cluster counts. Check robustness to sampling and parameter choices.
- Corresponding MD atoms and chains must genuinely match. Similarity on
  aligned PC coordinates alone cannot establish kinetics or stability.
- CSV features need comparable units or a **single consistent** scaling
  transformation applied across systems before running this code.
- Exact medoids use pairwise distance sums and can be slow for large clusters.

## Cite the method and implementation

1. Campello, R. J. G. B.; Moulavi, D.; Sander, J. **Density-Based Clustering
   Based on Hierarchical Density Estimates.** In *Advances in Knowledge
   Discovery and Data Mining* (PAKDD 2013), pp. 160–172.
   [doi:10.1007/978-3-642-37456-2_14](https://doi.org/10.1007/978-3-642-37456-2_14).
2. Campello, R. J. G. B.; Moulavi, D.; Zimek, A.; Sander, J. **Hierarchical
   Density Estimates for Data Clustering, Visualization, and Outlier
   Detection.** *ACM Transactions on Knowledge Discovery from Data* **10**
   (1), Article 5 (2015).
   [doi:10.1145/2733381](https://doi.org/10.1145/2733381).
3. McInnes, L.; Healy, J.; Astels, S. **hdbscan: Hierarchical density based
   clustering.** *Journal of Open Source Software* **2** (11), 205 (2017).
   [doi:10.21105/joss.00205](https://doi.org/10.21105/joss.00205).

The first two are by the original HDBSCAN method authors; the third covers
the Python implementation used in this repository. The package's
[algorithm guide](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)
explains the density hierarchy with diagrams.
