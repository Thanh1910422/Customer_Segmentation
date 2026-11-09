import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score
)


def evaluate_clusters(
    X_pca, min_k=4, max_k=8, random_state=42, min_cluster_share=0.05
):
    """Quét K và chọn theo thứ hạng đa chỉ số, không chỉ Silhouette."""
    n_samples = len(X_pca)
    if n_samples < 4:
        raise ValueError("Cần ít nhất 3 khách hàng để đánh giá.")
    max_k = min(max_k, n_samples - 1)

    results = {
        "k": [],
        "inertia": [],
        "silhouette": [],
        "davies_bouldin": [],
        "calinski_harabasz": [],
        "min_cluster_share": [],
        "mean_metric_rank": [],
        "eligible": []
    }
    for k in range(min_k, max_k + 1):
        model = KMeans(
            n_clusters=k, n_init=50,init="k-means++",max_iter=500, random_state=random_state
        )
        labels = model.fit_predict(X_pca)
        if np.unique(labels).size < 2:
            continue
        results["k"].append(k)
        results["inertia"].append(model.inertia_)
        results["silhouette"].append(
            silhouette_score(
                X_pca, labels, sample_size=min(3000, n_samples),
                random_state=42
            )
        )
        results["davies_bouldin"].append(
            davies_bouldin_score(X_pca, labels)
        )
        results["calinski_harabasz"].append(
            calinski_harabasz_score(X_pca, labels)
        )
        results["min_cluster_share"].append(
            float(np.bincount(labels).min() / n_samples)
        )
        print(
            f"K={k}: Inertia={model.inertia_:.2f}, "
            f"Silhouette={results['silhouette'][-1]:.4f}, "
            f"Davies-Bouldin={results['davies_bouldin'][-1]:.4f}, "
            f"Calinski-Harabasz={results['calinski_harabasz'][-1]:.2f}"
        )
    if not results["k"]:
        raise ValueError("Không tìm được kết quả K-Means hợp lệ.")

    silhouette_ranks = pd.Series(results["silhouette"]).rank(
        ascending=False, method="average"
    )
    db_ranks = pd.Series(results["davies_bouldin"]).rank(
        ascending=True, method="average"
    )
    ch_ranks = pd.Series(results["calinski_harabasz"]).rank(
        ascending=False, method="average"
    )
    results["mean_metric_rank"] = (
        (silhouette_ranks + db_ranks + ch_ranks) / 3
    ).tolist()
    results["eligible"] = [
        share >= min_cluster_share
        for share in results["min_cluster_share"]
    ]
    eligible_indices = [
        index for index, eligible in enumerate(results["eligible"])
        if eligible
    ]
    if not eligible_indices:
        eligible_indices = list(range(len(results["k"])))
        print(
            "Cảnh báo: không có K nào đạt tỷ lệ cụm tối thiểu; "
            "chọn trong toàn bộ các nghiệm."
        )
    best_index = min(
        eligible_indices,
        key=lambda index: (
            results["mean_metric_rank"][index],
            results["k"][index]
        )
    )
    results["best_k"] = results["k"][best_index]
    results["min_cluster_share_threshold"] = min_cluster_share
    print(
        "Chọn K theo thứ hạng trung bình Silhouette/DB/CH "
        f"(K >= {min_k}, tỷ lệ cụm nhỏ nhất >= "
        f"{min_cluster_share:.0%}): K={results['best_k']}."
    )
    return results


def train_kmeans(X_pca, results=None, n_clusters=None, random_state=42):
    """Huấn luyện K-Means với K thủ công hoặc tối ưu theo Silhouette."""
    if n_clusters is None:
        if results is None or not results.get("k"):
            raise ValueError("Cần kết quả quét K để tự chọn số cụm.")
        n_clusters = results["best_k"]
    if len(X_pca) < n_clusters:
        raise ValueError("Số khách hàng phải >= số cụm.")

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=50,
        max_iter=500,
        random_state=random_state
        )
    labels = model.fit_predict(X_pca)
    return model, labels
