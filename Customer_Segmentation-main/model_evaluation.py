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
        "Silhouette": silhouette_score(
            X_pca,
            labels,
            sample_size=min(3000, len(X_pca)),
            random_state=42
        ),
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
    """
    Kiểm tra độ ổn định bằng Subsampling.
    Mỗi lần huấn luyện sử dụng 80% dữ liệu,
    sau đó dự đoán trên toàn bộ dữ liệu để so sánh ARI.
    """

    label_runs = []

    X_array = np.asarray(X_pca)
    n_samples = len(X_array)

    if n_samples < n_clusters * 2:
        raise ValueError("Dữ liệu không đủ để kiểm tra ổn định.")

    rng = np.random.RandomState(42)

    for random_state in range(20):

        # Lấy ngẫu nhiên 80% dữ liệu
        sample_size = int(n_samples * 0.8)

        sample_indices = rng.choice(
            n_samples,
            size=sample_size,
            replace=False
        )

        X_sample = X_array[sample_indices]

        if model_type == "kmeans":

            model = KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=50,
                max_iter=500,
                random_state=random_state
            )

            model.fit(X_sample)

            # Dự đoán trên toàn bộ dữ liệu
            labels = model.predict(X_array)

        elif model_type == "gmm":

            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type=covariance_type,
                reg_covar=1e-4,
                n_init=10,
                max_iter=500,
                init_params="kmeans",
                random_state=random_state
            )

            model.fit(X_sample)

            if not model.converged_:
                print(
                    f"Cảnh báo: GMM seed={random_state} "
                    "không hội tụ."
                )
                continue

            labels = model.predict(X_array)

        else:
            raise ValueError(
                f"Loại mô hình không hỗ trợ: {model_type}"
            )

        label_runs.append(labels)

    if len(label_runs) < 2:
        raise RuntimeError(
            "Không đủ lần chạy hợp lệ để tính ARI."
        )

    mean_ari, std_ari = _stability_ari(label_runs)

    print(
        f"Độ ổn định {model_type}: "
        f"ARI = {mean_ari:.4f} ± {std_ari:.4f} "
        f"({len(label_runs)} lần Subsampling)."
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
