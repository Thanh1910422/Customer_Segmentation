from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score
)
from sklearn.mixture import GaussianMixture


def evaluate_model(X_pca, labels, model_name):
    """Tính ba chỉ số phân cụm trên cùng không gian PCA."""
    n_clusters = np.unique(labels).size
    if n_clusters < 2 or n_clusters >= len(X_pca):
        raise ValueError(f"{model_name} tạo số cụm không hợp lệ.")
    return {
        "Model": model_name,
        "Number_of_Clusters": int(n_clusters),
        "Silhouette": silhouette_score(X_pca, labels),
        "Davies_Bouldin": davies_bouldin_score(X_pca, labels),
        "Calinski_Harabasz": calinski_harabasz_score(X_pca, labels)
    }


def _stability_ari(scores):
    """Tính trung bình và độ lệch chuẩn ARI giữa mọi cặp lần chạy."""
    pair_scores = [
        adjusted_rand_score(scores[left], scores[right])
        for left, right in combinations(range(len(scores)), 2)
    ]
    return float(np.mean(pair_scores)), float(np.std(pair_scores))


def evaluate_stability(
    X_pca, model_type, n_clusters, covariance_type=None
):
    """Chạy mô hình với 10 seed xác định và tính độ ổn định ARI."""
    label_runs = []
    for random_state in range(42, 52):
        if model_type == "kmeans":
            model = KMeans(
                n_clusters=n_clusters, n_init=20,
                random_state=random_state
            )
            labels = model.fit_predict(X_pca)
        elif model_type == "gmm":
            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type=covariance_type,
                reg_covar=1e-3,
                n_init=10,
                init_params="kmeans",
                random_state=random_state
            )
            model.fit(X_pca)
            if not model.converged_:
                raise RuntimeError(
                    f"GMM seed={random_state} không hội tụ."
                )
            labels = model.predict(X_pca)
        else:
            raise ValueError(f"Loại mô hình không hỗ trợ: {model_type}")
        label_runs.append(labels)
    mean_ari, std_ari = _stability_ari(label_runs)
    print(
        f"Độ ổn định {model_type}: ARI = "
        f"{mean_ari:.4f} ± {std_ari:.4f} (10 lần chạy)."
    )
    return mean_ari, std_ari


def compare_models(
    X_pca,
    kmeans_labels,
    gmm_labels,
    kmeans_stability,
    gmm_stability
):
    """So sánh chỉ số và độ ổn định; đồng thuận chỉ có ý nghĩa khi K bằng nhau."""
    same_cluster_count = (
        np.unique(kmeans_labels).size == np.unique(gmm_labels).size
    )
    agreement_ari = (
        adjusted_rand_score(kmeans_labels, gmm_labels)
        if same_cluster_count else np.nan
    )
    agreement_nmi = (
        normalized_mutual_info_score(kmeans_labels, gmm_labels)
        if same_cluster_count else np.nan
    )
    rows = [
        evaluate_model(X_pca, kmeans_labels, "K-Means"),
        evaluate_model(X_pca, gmm_labels, "GMM")
    ]
    rows[0].update({
        "Stability_ARI_Mean": kmeans_stability[0],
        "Stability_ARI_Std": kmeans_stability[1],
        "Agreement_ARI": agreement_ari,
        "Agreement_NMI": agreement_nmi
    })
    rows[1].update({
        "Stability_ARI_Mean": gmm_stability[0],
        "Stability_ARI_Std": gmm_stability[1],
        "Agreement_ARI": agreement_ari,
        "Agreement_NMI": agreement_nmi
    })
    comparison = pd.DataFrame(rows)
    print("\nBẢNG SO SÁNH MÔ HÌNH (cùng X_pca)")
    print(comparison.to_string(index=False))
    if same_cluster_count:
        print(
            f"Đồng thuận K-Means/GMM cùng K: ARI={agreement_ari:.4f}, "
            f"NMI={agreement_nmi:.4f}"
        )
    else:
        print(
            "Bỏ qua ARI/NMI đồng thuận: hai mô hình có số cụm khác nhau."
        )
    return comparison
