import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score
)
from sklearn.mixture import GaussianMixture


COVARIANCE_TYPES = (
    "full",
    "tied",
    "diag",
    "spherical"
)
def evaluate_gmm_components(X_pca, min_components=2, max_components=12):
    """Quét BIC/AIC đến K=12 và nhận diện cực tiểu nội bộ."""
    n_samples = len(X_pca)
    if n_samples < 3:
        raise ValueError("Cần ít nhất 3 khách hàng để đánh giá.")
    max_components = min(max_components, n_samples - 1)

    records = []
    for covariance_type in COVARIANCE_TYPES:
        for n_components in range(min_components, max_components + 1):
            model = GaussianMixture(
                 n_components=n_components,
                covariance_type=covariance_type,
                reg_covar=1e-4,
                n_init=10,
                max_iter=500,
                init_params="kmeans",
                tol=1e-3,
                random_state=42
            )
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(X_pca)
                bic = float(model.bic(X_pca))
                aic = float(model.aic(X_pca))
                converged = bool(model.converged_)
                labels = model.predict(X_pca) if converged else None

                cluster_counts = (
                    np.bincount(labels, minlength=n_components)
                    if labels is not None
                    else np.zeros(n_components, dtype=int)
                )

                valid = (
                    converged
                    and np.isfinite(bic)
                    and np.isfinite(aic)
                    and np.all(cluster_counts > 0)
                    and np.min(model.weights_) >= 1e-4
                )
                if caught and not converged:
                    print(
                        f"Cảnh báo GMM {covariance_type}, "
                        f"K={n_components}: không hội tụ."
                    )
                if converged and np.min(model.weights_) < 1e-4:
                    print(
                        f"Bỏ GMM {covariance_type}, K={n_components}: "
                        "có thành phần gần như rỗng."
                    )
                labels = model.predict(X_pca) if valid else None
                silhouette = (
                    silhouette_score(
                        X_pca, labels,
                        sample_size=min(2000, n_samples),
                        random_state=42
                    )
                    if valid else np.nan
                )
                davies_bouldin = (
                    davies_bouldin_score(X_pca, labels)
                    if valid else np.nan
                )
                calinski_harabasz = (
                    calinski_harabasz_score(X_pca, labels)
                    if valid else np.nan
                )
                error = ""
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                bic = np.nan
                aic = np.nan
                silhouette = np.nan
                davies_bouldin = np.nan
                calinski_harabasz = np.nan
                converged = False
                valid = False
                error = str(exc)
                print(
                    f"Bỏ GMM {covariance_type}, K={n_components}: {error}"
                )

            records.append({
                "n_components": n_components,
                "covariance_type": covariance_type,
                "bic": bic,
                "aic": aic,
                "silhouette": silhouette,
                "davies_bouldin": davies_bouldin,
                "calinski_harabasz": calinski_harabasz,
                "converged": converged,
                "valid": valid,
                "error": error
            })
            status = "hợp lệ" if valid else "không hợp lệ"
            print(
                f"GMM {covariance_type}, K={n_components}: "
                f"BIC={bic:.2f}, AIC={aic:.2f}, "
                f"Silhouette={silhouette:.4f} ({status})"
            )

    valid_records = [record for record in records if record["valid"]]
    if not valid_records:
        raise RuntimeError("Không có cấu hình GMM nào hội tụ hợp lệ.")
    best = min(valid_records, key=lambda record: record["bic"])
    local_minima = []
    for covariance_type in COVARIANCE_TYPES:
        curve = sorted(
            (
                record for record in valid_records
                if record["covariance_type"] == covariance_type
            ),
            key=lambda record: record["n_components"]
        )
        for index in range(1, len(curve) - 1):
            current = curve[index]
            if (
                current["bic"] < curve[index - 1]["bic"]
                and current["bic"] < curve[index + 1]["bic"]
            ):
                local_minima.append({
                    "n_components": current["n_components"],
                    "covariance_type": covariance_type,
                    "bic": current["bic"]
                })
    return {
        "records": records,
        "local_minima": local_minima,
        "best_config": {
            "n_components": best["n_components"],
            "covariance_type": best["covariance_type"],
            "bic": best["bic"],
            "aic": best["aic"],
            "at_search_boundary": best["n_components"] == max_components
        }
    }


def select_gmm_config_for_k(results, n_components):
    """Chọn covariance có Silhouette tốt nhất tại K cố định."""
    candidates = [
        record for record in results["records"]
        if record["n_components"] == n_components and record["valid"]
    ]
    if not candidates:
        raise RuntimeError(
            f"Không có cấu hình GMM hợp lệ tại K={n_components}."
        )
    selected = max(
        candidates,
        key=lambda record: (record["silhouette"], -record["bic"])
    )
    bic_best = min(candidates, key=lambda record: record["bic"])
    return {
        "n_components": selected["n_components"],
        "covariance_type": selected["covariance_type"],
        "bic": selected["bic"],
        "aic": selected["aic"],
        "silhouette": selected["silhouette"],
        "selection_basis": (
            f"Silhouette cao nhất tại K={n_components}; "
            "BIC dùng để phá hòa"
        ),
        "bic_best_covariance_type": bic_best["covariance_type"],
        "bic_best_at_k": bic_best["bic"]
    }


def train_gmm(
    X_pca, n_components, covariance_type, random_state=42
):
    """Huấn luyện GMM và trả về mô hình, nhãn, xác suất và độ chắc chắn."""
    if len(X_pca) < n_components:
        raise ValueError("Số khách hàng phải >= số thành phần.")
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=1e-4,
        n_init=10,
        max_iter=500,
        init_params="kmeans",
        tol=1e-3,
        random_state=random_state
    )
    model.fit(X_pca)
    if not model.converged_:
        raise RuntimeError(
            f"GMM ({covariance_type}, K={n_components}) không hội tụ."
        )
    labels = model.predict(X_pca)
    probabilities = model.predict_proba(X_pca)
    confidence = probabilities.max(axis=1)
    return model, labels, probabilities, confidence
