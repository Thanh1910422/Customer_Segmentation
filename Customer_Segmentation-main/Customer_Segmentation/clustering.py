
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


def evaluate_clusters(X_scaled, max_k=10):
    results = {
        "k": [],
        "inertia": [],
        "silhouette": [],
        "davies_bouldin": [],
        "calinski_harabasz": []
    }

    max_k = min(max_k, len(X_scaled) - 1)

    if max_k < 2:
        raise ValueError("Cần ít nhất 3 khách hàng để đánh giá.")

    for k in range(2, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(X_scaled)

        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(
            X_scaled, labels
        )
        calinski_harabasz = calinski_harabasz_score(
            X_scaled, labels
        )

        results["k"].append(k)
        results["inertia"].append(model.inertia_)
        results["silhouette"].append(silhouette)
        results["davies_bouldin"].append(davies_bouldin)
        results["calinski_harabasz"].append(calinski_harabasz)

        print(
            f"K={k}: "
            f"Inertia={model.inertia_:.2f}, "
            f"Silhouette={silhouette:.4f}, "
            f"Davies-Bouldin={davies_bouldin:.4f}, "
            f"Calinski-Harabasz={calinski_harabasz:.2f}"
        )

    return results


def train_kmeans(X_scaled, n_clusters=4):
    if len(X_scaled) < n_clusters:
        raise ValueError(
            "Số khách hàng phải lớn hơn hoặc bằng số cụm."
        )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X_scaled)

    return model, labels
