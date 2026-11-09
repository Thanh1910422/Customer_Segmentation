import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_output(output_dir):
    """Tạo thư mục đích chứa biểu đồ."""
    os.makedirs(output_dir, exist_ok=True)


def plot_elbow(results, output_dir, selected_k):
    """Vẽ Inertia theo K và đánh dấu K được chọn."""
    _prepare_output(output_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["k"], results["inertia"], marker="o")
    ax.axvline(selected_k, color="tab:red", linestyle="--",
               label=f"K được chọn = {selected_k}")
    ax.set(title="Elbow Method - K-Means", xlabel="Số cụm (K)",
           ylabel="Inertia")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "elbow_method.png"), dpi=150)
    plt.close(fig)


def plot_silhouette(results, output_dir, selected_k):
    """Vẽ Silhouette theo K và đánh dấu K tối ưu."""
    _prepare_output(output_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["k"], results["silhouette"], marker="o")
    ax.scatter(
        [selected_k],
        [results["silhouette"][results["k"].index(selected_k)]],
        color="tab:red", zorder=3, label=f"K được chọn = {selected_k}"
    )
    ax.set(title="Silhouette Score - K-Means", xlabel="Số cụm (K)",
           ylabel="Silhouette")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "silhouette_score.png"), dpi=150)
    plt.close(fig)


def plot_gmm_information_criteria(results, output_dir, best_config):
    """Vẽ tiêu chí theo K, gồm cấu hình cùng K và cực tiểu trong dải."""
    _prepare_output(output_dir)
    records = pd.DataFrame(results["records"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for covariance_type, group in records.groupby("covariance_type"):
        valid = group[group["valid"]]
        axes[0].plot(
            valid["n_components"], valid["bic"], marker="o",
            label=covariance_type
        )
        axes[1].plot(
            valid["n_components"], valid["aic"], marker="o",
            label=covariance_type
        )
    for ax, metric in zip(axes, ("bic", "aic")):
        scan_min = records.loc[records[metric].idxmin()]
        ax.scatter(
            [best_config["n_components"]], [best_config[metric]],
            color="red", marker="*", s=180, zorder=5,
            label=f"GMM cùng K={best_config['n_components']}"
        )
        ax.scatter(
            [scan_min["n_components"]], [scan_min[metric]],
            color="black", marker="x", s=90, zorder=6,
            label=f"Thấp nhất trong dải: K={int(scan_min['n_components'])}"
        )
        ax.set_xlabel("Số thành phần")
        ax.set_ylabel(metric.upper() + " (thấp hơn tốt hơn)")
        ax.grid(True)
        ax.legend()
    axes[0].set_title("BIC theo covariance_type")
    axes[1].set_title("AIC theo covariance_type")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "gmm_aic_bic.png"), dpi=150)
    plt.close(fig)


def _plot_discrete_clusters(ax, x_values, y_values, labels, title):
    """Vẽ nhãn cụm rời rạc, không dùng colorbar liên tục."""
    unique_labels = np.unique(labels)
    colors = plt.get_cmap("tab10")
    for color_index, cluster in enumerate(unique_labels):
        mask = labels == cluster
        ax.scatter(
            x_values[mask], y_values[mask], s=8, alpha=0.6,
            color=colors(color_index % 10), label=f"Cụm {cluster}"
        )
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(title="Nhãn cụm", markerscale=2, fontsize=8)


def plot_pca_cluster_scatter(X_pca, kmeans_labels, gmm_labels, output_dir):
    """Vẽ K-Means và GMM cạnh nhau trên PC1/PC2."""
    _prepare_output(output_dir)
    variance = X_pca.attrs["explained_variance_ratio_"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, labels, title in (
        (
            axes[0], kmeans_labels,
            f"K-Means (K={np.unique(kmeans_labels).size})"
        ),
        (
            axes[1], gmm_labels,
            f"GMM (K={np.unique(gmm_labels).size})"
        )
    ):
        _plot_discrete_clusters(
            ax, X_pca.iloc[:, 0].to_numpy(),
            X_pca.iloc[:, 1].to_numpy(), labels, title
        )
        ax.set_xlabel(f"PC1 ({variance[0]:.1%} phương sai)")
        ax.set_ylabel(f"PC2 ({variance[1]:.1%} phương sai)")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "clusters_pc1_pc2.png"), dpi=150
    )
    plt.close(fig)


def plot_pc1_pc3(X_pca, kmeans_labels, gmm_labels, output_dir):
    """Vẽ PC1/PC3 để kiểm tra phân tách theo chiều PCA khác."""
    _prepare_output(output_dir)
    if X_pca.shape[1] < 3:
        print("Cảnh báo: PCA có dưới 3 thành phần; bỏ qua biểu đồ PC1/PC3.")
        return
    variance = X_pca.attrs["explained_variance_ratio_"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, labels, title in (
        (
            axes[0], kmeans_labels,
            f"K-Means (K={np.unique(kmeans_labels).size})"
        ),
        (
            axes[1], gmm_labels,
            f"GMM (K={np.unique(gmm_labels).size})"
        )
    ):
        _plot_discrete_clusters(
            ax, X_pca.iloc[:, 0].to_numpy(),
            X_pca.iloc[:, 2].to_numpy(), labels, title
        )
        ax.set_xlabel(f"PC1 ({variance[0]:.1%} phương sai)")
        ax.set_ylabel(f"PC3 ({variance[2]:.1%} phương sai)")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "clusters_pc1_pc3.png"), dpi=150
    )
    plt.close(fig)


def plot_cluster_sizes(kmeans_labels, gmm_labels, output_dir):
    """So sánh phân bố kích thước cụm của hai mô hình."""
    _prepare_output(output_dir)
    counts = pd.concat(
        [
            pd.Series(kmeans_labels).value_counts().rename(
                f"K-Means (K={np.unique(kmeans_labels).size})"
            ),
            pd.Series(gmm_labels).value_counts().rename(
                f"GMM (K={np.unique(gmm_labels).size})"
            )
        ],
        axis=1
    ).fillna(0).sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set(title="Kích thước cụm", xlabel="Nhãn cụm",
           ylabel="Số khách hàng")
    ax.legend(title="Mô hình")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cluster_sizes.png"), dpi=150)
    plt.close(fig)


def plot_cluster_heatmaps(X_scaled, kmeans_labels, gmm_labels, output_dir):
    """Vẽ heatmap trung bình biến chuẩn hóa theo cụm."""
    _prepare_output(output_dir)
    fig, axes = plt.subplots(2, 1, figsize=(max(11, X_scaled.shape[1]), 9))
    for ax, labels, title in (
        (
            axes[0], kmeans_labels,
            f"K-Means (K={np.unique(kmeans_labels).size})"
        ),
        (
            axes[1], gmm_labels,
            f"GMM (K={np.unique(gmm_labels).size})"
        )
    ):
        means = X_scaled.assign(Cluster=labels).groupby("Cluster").mean()
        heatmap = ax.imshow(
            means.to_numpy(), cmap="coolwarm", vmin=-2, vmax=2,
            aspect="auto"
        )
        ax.set_xticks(np.arange(len(means.columns)))
        ax.set_xticklabels(means.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(means.index)))
        ax.set_yticklabels(means.index)
        fig.colorbar(
            heatmap, ax=ax, label="Trung bình chuẩn hóa", shrink=0.8
        )
        ax.set_title(f"Đặc trưng trung bình theo cụm - {title}")
        ax.set_xlabel("Biến")
        ax.set_ylabel("Cụm")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "cluster_feature_heatmaps.png"), dpi=150
    )
    plt.close(fig)


def plot_umap_if_available(X_scaled, kmeans_labels, gmm_labels, output_dir):
    """Vẽ UMAP nếu umap-learn đã có, không cài thêm phụ thuộc."""
    _prepare_output(output_dir)
    try:
        import umap
    except ImportError:
        print("Cảnh báo: chưa cài umap-learn; bỏ qua biểu đồ UMAP.")
        return

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, labels, title in (
        (
            axes[0], kmeans_labels,
            f"K-Means (K={np.unique(kmeans_labels).size})"
        ),
        (
            axes[1], gmm_labels,
            f"GMM (K={np.unique(gmm_labels).size})"
        )
    ):
        _plot_discrete_clusters(
            ax, embedding[:, 0], embedding[:, 1], labels, title
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "clusters_umap.png"), dpi=150)
    plt.close(fig)
