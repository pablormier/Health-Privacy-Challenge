import umap
import os
import numpy as np
import scanpy as sc
import scipy.stats as stats
import scipy.sparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import adjusted_rand_score, roc_auc_score, jaccard_score
from scib.metrics import ilisi_graph
import celltypist
from scipy.sparse import issparse

def filter_low_quality_cells_and_genes(adata, min_counts=10, min_cells=3):
    """
    Filters cells and genes based on minimum counts.
    Uses Scanpy’s built-in filtering functions (which are sparse-aware).
    """
    adata = adata.copy()
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    return adata

def get_dense_column(adata, i):
    """
    Returns the i-th column of adata.X as a dense vector.
    This avoids converting the entire matrix to dense at once.
    """
    X = adata.X
    if issparse(X):
        return X[:, i].toarray().ravel()
    else:
        return np.array(X[:, i]).ravel()

def check_for_inf_nan(adata, label):
    """
    Checks for NaN/Inf values in adata.X without converting the whole matrix.
    """
    X = adata.X
    if issparse(X):
        data = X.data
    else:
        data = np.array(X)
    print(f"Checking {label} dataset:")
    print(f"NaNs? {np.isnan(data).any()}")
    print(f"Infs? {np.isinf(data).any()}")
    print(f"Min: {data.min()}, Max: {data.max()}\n")

def check_missing_genes(real_data, synthetic_data):
    """
    Compares gene names between real and synthetic datasets.
    """
    real_genes = set(real_data.var_names)
    synthetic_genes = set(synthetic_data.var_names)
    missing_in_real = synthetic_genes - real_genes
    missing_in_synthetic = real_genes - synthetic_genes

    print(f"Genes in synthetic but not in real: {len(missing_in_real)}")
    print(f"Genes in real but not in synthetic: {len(missing_in_synthetic)}")
    print(f"Example missing in real: {list(missing_in_real)[:10]}")
    print(f"Example missing in synthetic: {list(missing_in_synthetic)[:10]}")
    print(f"real_data.var_names dtype: {real_data.var_names.dtype}")
    print(f"synthetic_data.var_names dtype: {synthetic_data.var_names.dtype}")

class Statistics:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def compute_scc(self, real_data, synthetic_data, n_hvgs=5000):
        """
        Computes the mean Spearman correlation across highly variable genes (HVGs)
        between the real and synthetic datasets. Instead of converting the whole
        expression matrix to dense, each gene column is converted on the fly.
        """
        np.random.seed(self.random_seed)
        check_missing_genes(real_data, synthetic_data)
        # Align genes using the gene names from synthetic_data
        common_genes = synthetic_data.var_names
        real_data = real_data[:, common_genes]
        synthetic_data = synthetic_data[:, common_genes]

        check_for_inf_nan(real_data, "Real")
        check_for_inf_nan(synthetic_data, "Synthetic")

        # Normalize and log-transform both datasets
        sc.pp.normalize_total(real_data, target_sum=1e4)
        sc.pp.log1p(real_data)
        sc.pp.normalize_total(synthetic_data, target_sum=1e4)
        sc.pp.log1p(synthetic_data)

        check_for_inf_nan(real_data, "Real")
        check_for_inf_nan(synthetic_data, "Synthetic")

        # Identify HVGs using the combined dataset
        combined_adata = real_data.concatenate(synthetic_data)
        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)

        # Subset to HVGs
        hvgs = combined_adata.var["highly_variable"]
        real_hvg = real_data[:, hvgs]
        synth_hvg = synthetic_data[:, hvgs]

        # Compute Spearman correlation gene-by-gene
        scc_values = []
        for i in range(real_hvg.n_vars):
            real_vec = get_dense_column(real_hvg, i)
            synth_vec = get_dense_column(synth_hvg, i)
            corr, _ = stats.spearmanr(real_vec, synth_vec, nan_policy='omit')
            scc_values.append(corr)
        scc_values = np.array(scc_values)
        return np.nanmean(scc_values) if not np.all(np.isnan(scc_values)) else np.nan

    def compute_mmd_optimized(self, real_data, synthetic_data, sample_size=20000,
                               n_pca=50, gamma=1.0, n_hvgs=5000):
        """
        Computes the Maximum Mean Discrepancy (MMD) between the two datasets.
        Uses a sparse-aware subsampling and PCA approach. If the HVG data remains sparse,
        TruncatedSVD is used instead of the standard PCA.
        """
        np.random.seed(self.random_seed)
        # Align genes using synthetic_data's ordering
        common_genes = synthetic_data.var_names
        real_data = real_data[:, common_genes]
        synthetic_data = synthetic_data[:, common_genes]

        combined_adata = real_data.concatenate(synthetic_data)
        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)

        hvgs = combined_adata.var["highly_variable"]
        real_hvg = real_data[:, hvgs]
        synth_hvg = synthetic_data[:, hvgs]

        n_real = real_hvg.n_obs
        n_synth = synth_hvg.n_obs

        real_idx = np.random.choice(n_real, min(sample_size, n_real), replace=False)
        synth_idx = np.random.choice(n_synth, min(sample_size, n_synth), replace=False)

        # Process sparse or dense data accordingly
        if issparse(real_hvg.X):
            real_sample = real_hvg.X[real_idx]
            synth_sample = synth_hvg.X[synth_idx]
            from scipy.sparse import vstack
            combined_sample = vstack([real_sample, synth_sample])
            pca_model = TruncatedSVD(n_components=n_pca, random_state=self.random_seed)
            combined_pca = pca_model.fit_transform(combined_sample)
        else:
            real_sample = real_hvg.X[real_idx]
            synth_sample = synth_hvg.X[synth_idx]
            combined_sample = np.vstack([real_sample, synth_sample])
            pca_model = PCA(n_components=n_pca, random_state=self.random_seed)
            combined_pca = pca_model.fit_transform(combined_sample)

        real_pca = combined_pca[:len(real_sample)]
        synth_pca = combined_pca[len(real_sample):]

        K_xx = rbf_kernel(real_pca, real_pca, gamma=gamma).mean()
        K_yy = rbf_kernel(synth_pca, synth_pca, gamma=gamma).mean()
        K_xy = rbf_kernel(real_pca, synth_pca, gamma=gamma).mean()

        return K_xx + K_yy - 2 * K_xy

    def compute_lisi(self, real_data, synthetic_data, n_hvgs=5000):
        """
        Computes the Local Inverse Simpson’s Index (LISI) to measure mixing
        of real and synthetic cells in a shared low-dimensional space.
        """
        np.random.seed(self.random_seed)
        common_genes = synthetic_data.var_names
        real_data = real_data[:, common_genes]
        synthetic_data = synthetic_data[:, common_genes]
        combined_adata = real_data.concatenate(
            synthetic_data, batch_key="source", batch_categories=["real", "synthetic"]
        )
        # Create a numeric batch label (0 = real, 1 = synthetic)
        combined_adata.obs["batch"] = (combined_adata.obs["source"] == "synthetic").astype(int)

        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)
        combined_adata = combined_adata[:, combined_adata.var['highly_variable']]

        sc.pp.pca(combined_adata, n_comps=100, random_state=self.random_seed)
        sc.pp.neighbors(combined_adata, n_neighbors=10, method='umap')

        return ilisi_graph(combined_adata, batch_key="batch", type_="knn")

    def compute_ari(self, real_data, synthetic_data, cell_type_col, n_hvgs=5000):
        """
        Computes the Adjusted Rand Index (ARI) to measure clustering consistency
        between real and synthetic data. Clusters are obtained via Scanpy's Louvain.
        """
        np.random.seed(self.random_seed)
        common_genes = synthetic_data.var_names
        real_data = real_data[:, common_genes]
        synthetic_data = synthetic_data[:, common_genes]
        combined_adata = real_data.concatenate(
            synthetic_data, batch_key="source", batch_categories=["real", "synthetic"]
        )

        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)
        combined_adata = combined_adata[:, combined_adata.var['highly_variable']]

        sc.pp.pca(combined_adata, n_comps=100, random_state=self.random_seed)
        sc.pp.neighbors(combined_adata, n_neighbors=10, method='umap')
        sc.tl.louvain(combined_adata)

        # Convert Louvain clusters to numerical labels
        combined_adata.obs["louvain"] = combined_adata.obs["louvain"].astype("category").cat.codes
        real_clusters = combined_adata.obs.loc[combined_adata.obs["source"] == "real", "louvain"].values
        synthetic_clusters = combined_adata.obs.loc[combined_adata.obs["source"] == "synthetic", "louvain"].values
        ari_real_vs_syn = adjusted_rand_score(real_clusters, synthetic_clusters)
        ari_gt_vs_comb = adjusted_rand_score(combined_adata.obs[cell_type_col], combined_adata.obs["louvain"])

        return ari_real_vs_syn, ari_gt_vs_comb

class VisualizeClassify:
    def __init__(self, sc_figures_dir, random_seed=42):
        self.random_seed = random_seed
        self.sc_figures_dir = sc_figures_dir
        np.random.seed(self.random_seed)

    def plot_umap(self, real_data, synthetic_data, n_hvgs=5000):
        """
        Creates and saves a UMAP plot of the combined real and synthetic data.
        """
        sc.settings.figdir = self.sc_figures_dir
        np.random.seed(self.random_seed)
        check_for_inf_nan(real_data, "Real")
        check_for_inf_nan(synthetic_data, "Synthetic")
        combined_adata = real_data.concatenate(
            synthetic_data, batch_key="source", batch_categories=["real", "synthetic"]
        )

        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)
        combined_adata = combined_adata[:, combined_adata.var['highly_variable']]

        sc.pp.pca(combined_adata, random_state=self.random_seed)
        sc.pp.neighbors(combined_adata)
        sc.tl.umap(combined_adata, random_state=self.random_seed)

        sc.pl.umap(combined_adata,
                   color=["source"],
                   title="UMAP of Real vs Synthetic Data",
                   save=f"syn_test_PCA_HVG={n_hvgs}.png")

    def celltypist_classification(self, real_data_test, synthetic_data, celltypist_model, n_hvgs=5000):
        """
        Uses a CellTypist model to annotate cells from both datasets and then compares
        the predicted labels via ARI and Jaccard scores.
        """
        np.random.seed(self.random_seed)
        combined_adata = real_data_test.concatenate(synthetic_data)
        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)

        # Normalize and log-transform each dataset individually
        sc.pp.normalize_total(real_data_test, target_sum=1e4)
        sc.pp.log1p(real_data_test)
        sc.pp.normalize_total(synthetic_data, target_sum=1e4)
        sc.pp.log1p(synthetic_data)

        # Subset both datasets to HVGs
        real_data_test = real_data_test[:, combined_adata.var['highly_variable']]
        synthetic_data = synthetic_data[:, combined_adata.var['highly_variable']]

        model = celltypist.models.Model.load(celltypist_model)
        real_predictions = celltypist.annotate(real_data_test, model=model)
        synthetic_predictions = celltypist.annotate(synthetic_data, model=model)

        real_labels = real_predictions.predicted_labels.values.ravel()
        synthetic_labels = synthetic_predictions.predicted_labels.values.ravel()

        ari_score = adjusted_rand_score(real_labels, synthetic_labels)

        lb = LabelBinarizer()
        real_onehot = lb.fit_transform(real_labels)
        synthetic_onehot = lb.transform(synthetic_labels)

        jaccard_scores = [
            jaccard_score(real_onehot[:, i], synthetic_onehot[:, i])
            for i in range(real_onehot.shape[1])
        ]
        jaccard = np.mean(jaccard_scores)

        return ari_score, jaccard

    def random_forest_eval(self, real_data, synthetic_data, n_hvgs=5000):
        """
        Evaluates how well a Random Forest can separate real vs. synthetic cells.
        After batch correction, the expression matrix is converted to dense only once.
        """
        np.random.seed(self.random_seed)
        real_data.obs["source"] = "real"
        synthetic_data.obs["source"] = "synthetic"

        combined_adata = real_data.concatenate(
            synthetic_data, batch_key="source", batch_categories=["real", "synthetic"]
        )

        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)
        combined_adata = combined_adata[:, combined_adata.var['highly_variable']]

        sc.pp.combat(combined_adata, key="source")

        # Convert to dense only at this final stage (necessary for RandomForest)
        X = combined_adata.X.A if hasattr(combined_adata.X, "A") else combined_adata.X
        y = (combined_adata.obs["source"] == "synthetic").astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=self.random_seed)

        rf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=self.random_seed)
        rf.fit(X_train, y_train)

        pred_probs = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, pred_probs)

        return auc, pred_probs
