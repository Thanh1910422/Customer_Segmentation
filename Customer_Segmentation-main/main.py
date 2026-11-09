from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocessing import load_and_preprocess
from gmm_model import (
    evaluate_gmm_components,
    select_gmm_config_for_k,
    train_gmm
)
from kmeans_model import evaluate_clusters, train_kmeans
from model_evaluation import (
    compare_models,
    evaluate_model,
    evaluate_stability
)
from visualization import (
    plot_cluster_heatmaps,
    plot_cluster_sizes,
    plot_elbow,
    plot_gmm_information_criteria,
    plot_pc1_pc3,
    plot_pca_cluster_scatter,
    plot_silhouette,
    plot_umap_if_available
)


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DATA_PATH = BASE_DIR / "CC GENERAL.csv"
KMEANS_K = 5
GMM_K = 5
CLUSTERING_PCA_COMPONENTS = 2

# Chỉ số lưu từ pipeline cũ để có đối chiếu trước/sau minh bạch.
BASELINE_ROWS = [
    {
        "Comparison_Phase": "Trước cải tiến",
        "Model": "K-Means (K=4)",
        "Number_of_Clusters": 4,
        "Silhouette": 0.2579515536192368,
        "Davies_Bouldin": 1.314266010364571,
        "Calinski_Harabasz": 3279.841089451172
    },
    {
        "Comparison_Phase": "Trước cải tiến",
        "Model": "GMM (K=4)",
        "Number_of_Clusters": 4,
        "Silhouette": 0.1090883349468052,
        "Davies_Bouldin": 2.7996447903534434,
        "Calinski_Harabasz": 994.6332073443157
    }
]


def save_cluster_profile(df, labels, output_path):
    """Lưu trung bình biến gốc, số lượng và tỷ lệ theo cụm."""
    numeric_columns = [
        column for column in df.select_dtypes(include=np.number)
        if column != "CUST_ID"
    ]
    profile_data = df[numeric_columns].copy()
    profile_data["Cluster"] = labels
    means = profile_data.groupby("Cluster").mean()
    counts = profile_data.groupby("Cluster").size().rename("Customer_Count")
    means.insert(0, "Customer_Count", counts)
    means.insert(1, "Percentage", counts / len(df) * 100)
    means.to_csv(output_path, encoding="utf-8-sig")


def print_cluster_insights(X_scaled, kmeans_labels, gmm_labels):
    """In 3–5 nhận xét về đặc điểm các cụm của hai mô hình."""
    print("\nNHẬN XÉT ĐẶC ĐIỂM CỤM")
    for model_name, labels, limit in (
        ("K-Means", kmeans_labels, 2),
        ("GMM", gmm_labels, 3)
    ):
        grouped = X_scaled.assign(Cluster=labels).groupby("Cluster").mean()
        shares = pd.Series(labels).value_counts(normalize=True)
        largest = shares.nlargest(limit).index
        for cluster in largest:
            row = grouped.loc[cluster].sort_values()
            lowest = row.index[0]
            highest = row.index[-1]
            print(
                f"- {model_name} cụm {cluster} "
                f"({shares[cluster]:.1%} khách): "
                f"cao tương đối ở {highest} ({row[highest]:+.2f} SD), "
                f"thấp ở {lowest} ({row[lowest]:+.2f} SD)."
            )


def select_clustering_components(X_pca):
    """Use the leading PCA dimensions as the clustering space."""
    if X_pca.shape[1] < CLUSTERING_PCA_COMPONENTS:
        raise ValueError("PCA cần ít nhất 2 thành phần để phân cụm.")
    explained_variance = X_pca.attrs.get("explained_variance_ratio_")
    if explained_variance is None:
        raise ValueError("Thiếu tỷ lệ phương sai giải thích của PCA.")
    selected = X_pca.iloc[:, :CLUSTERING_PCA_COMPONENTS].copy()
    selected.attrs["explained_variance_ratio_"] = explained_variance
    return selected


def main():
    """Chạy toàn bộ pipeline phân khúc và lưu bảng/biểu đồ."""
    print("=" * 72)
    print("PHÂN KHÚC KHÁCH HÀNG SỬ DỤNG THẺ TÍN DỤNG")
    print("=" * 72)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {DATA_PATH}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Tiền xử lý
    print("\n[1] ĐỌC DỮ LIỆU VÀ TIỀN XỬ LÝ")
    df_processed, X_scaled, X_pca, feature_names = load_and_preprocess(
        DATA_PATH
    )
    X_pca_all = X_pca
    X_pca = select_clustering_components(X_pca_all)
    retained_variance = X_pca.attrs["explained_variance_ratio_"][
        :CLUSTERING_PCA_COMPONENTS
    ].sum()
    print(
        f"Khách hàng={len(df_processed)}, biến={len(feature_names)}; "
        f"phân cụm trên PC1-PC{CLUSTERING_PCA_COMPONENTS}, "
        f"giữ {retained_variance:.2%} phương sai."
    )

    # 2. Quét K để tham khảo, sau đó huấn luyện K-Means với K cố định là 5.
    print("\n[2] QUÉT K-MEANS TRÊN X_pca")
    kmeans_results = evaluate_clusters(
        X_pca, min_k=2, max_k=10, min_cluster_share=0.05
    )
    kmeans_model, kmeans_labels = train_kmeans(
        X_pca, n_clusters=KMEANS_K
    )
    print(
        f"K-Means chạy với K={KMEANS_K} theo yêu cầu "
        f"(K được gợi ý bởi quét đa chỉ số: {kmeans_results['best_k']})."
    )

    print("\n[2a] KIỂM TRA ẢNH HƯỞNG CỦA BIẾN BỔ SUNG")
    _, _, X_pca_extra, extra_feature_names = (
        load_and_preprocess(
            DATA_PATH, include_additional_features=True
        )
    )
    X_pca_extra = select_clustering_components(X_pca_extra)
    _, extra_kmeans_labels = train_kmeans(
        X_pca_extra, n_clusters=KMEANS_K
    )
    feature_comparison = pd.DataFrame([
        {
            "Feature_Set": "Tập biến gọn",
            "Features": ", ".join(feature_names),
            **evaluate_model(X_pca, kmeans_labels, "K-Means"),
        },
        {
            "Feature_Set": "Tập biến gọn + PURCHASES_TRX, "
                           "MINIMUM_PAYMENTS, TENURE",
            "Features": ", ".join(extra_feature_names),
            **evaluate_model(
                X_pca_extra, extra_kmeans_labels, "K-Means"
            ),
        }
    ]).drop(columns=["Model"])
    feature_comparison.to_csv(
        RESULTS_DIR / "feature_set_experiment.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(
        feature_comparison[
            [
                "Feature_Set", "Number_of_Clusters", "Silhouette",
                "Davies_Bouldin", "Calinski_Harabasz"
            ]
        ].to_string(index=False)
    )
    print(
        "Biến bổ sung được kiểm tra nhưng không mặc định đưa vào mô hình; "
        "so sánh trên cùng K."
    )

    # 3. Quét cấu hình GMM; huấn luyện cùng K=5 với K-Means.
    print("\n[3] QUÉT GMM TRÊN X_pca")
    gmm_results = evaluate_gmm_components(
        X_pca, min_components=2, max_components=15
    )
    scan_minimum = gmm_results["best_config"]
    gmm_config = select_gmm_config_for_k(gmm_results, GMM_K)
    print(
        f"GMM chạy với K={GMM_K}: "
        f"covariance_type={gmm_config['covariance_type']}, "
        f"Silhouette={gmm_config['silhouette']:.4f}, "
        f"BIC={gmm_config['bic']:.2f}."
    )
    print(
        f"Tại K={GMM_K}, BIC thấp nhất thuộc "
        f"{gmm_config['bic_best_covariance_type']} "
        f"(BIC={gmm_config['bic_best_at_k']:.2f}); "
        "mô hình đối chiếu được chọn theo Silhouette ở cùng K."
    )
    print(
        f"BIC thấp nhất toàn dải khảo sát tại "
        f"K={scan_minimum['n_components']} "
        f"({scan_minimum['covariance_type']}, "
        f"BIC={scan_minimum['bic']:.2f})"
        + (" ở biên K=12." if scan_minimum["at_search_boundary"] else ".")
    )
    if gmm_results["local_minima"]:
        minima_text = "; ".join(
            f"K={item['n_components']} {item['covariance_type']} "
            f"(BIC={item['bic']:.2f})"
            for item in gmm_results["local_minima"]
        )
        print("Cực tiểu nội bộ BIC:", minima_text)
    else:
        print(
            "Không thấy cực tiểu nội bộ BIC trong K=2..15; "
            "không diễn giải giá trị K tại biên là số cụm tối ưu."
        )
    gmm_model, gmm_labels, probabilities, confidence = train_gmm(
        X_pca,
        n_components=GMM_K,
        covariance_type=gmm_config["covariance_type"]
    )

    # 4. Đánh giá chỉ số, độ ổn định và đồng thuận ở cùng K.
    print("\n[4] ĐÁNH GIÁ VÀ SO SÁNH MÔ HÌNH")
    kmeans_stability = evaluate_stability(
        X_pca, "kmeans", KMEANS_K
    )
    gmm_stability = evaluate_stability(
        X_pca,
        "gmm",
        GMM_K,
        covariance_type=gmm_config["covariance_type"]
    )
    comparison = compare_models(
        X_pca,
        kmeans_labels,
        gmm_labels,
        kmeans_stability,
        gmm_stability
    )

    previous = pd.DataFrame(BASELINE_ROWS)
    current = comparison.copy()
    current.insert(0, "Comparison_Phase", "Sau cải tiến")
    full_comparison = pd.concat(
        [previous, current], ignore_index=True, sort=False
    )
    full_comparison.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print("\nSO SÁNH CHỈ SỐ TRƯỚC / SAU")
    print(
        full_comparison[
            [
                "Comparison_Phase", "Model", "Number_of_Clusters",
                "Silhouette", "Davies_Bouldin", "Calinski_Harabasz"
            ]
        ].to_string(index=False)
    )
    if comparison.loc[0, "Silhouette"] < 0.35:
        print(
            "Lưu ý: Silhouette K-Means chưa đạt 0.35; "
            "cấu trúc cụm được báo cáo trung thực."
        )

    # 5. Lưu nhãn, độ chắc chắn, xác suất và hồ sơ cụm.
    print("\n[5] LƯU KẾT QUẢ VÀ HỒ SƠ CỤM")
    segments = df_processed.copy()
    segments["KMeans_Cluster"] = kmeans_labels
    segments["GMM_Cluster"] = gmm_labels
    segments["GMM_Confidence"] = confidence
    for component in range(probabilities.shape[1]):
        segments[f"GMM_Probability_{component}"] = probabilities[:, component]
    segments.to_csv(
        RESULTS_DIR / "customer_segments.csv",
        index=False,
        encoding="utf-8-sig"
    )
    save_cluster_profile(
        df_processed, kmeans_labels,
        RESULTS_DIR / "cluster_profiles_kmeans.csv"
    )
    save_cluster_profile(
        df_processed, gmm_labels,
        RESULTS_DIR / "cluster_profiles_gmm.csv"
    )
    pd.DataFrame(kmeans_results).drop(
        columns=["best_k"]
    ).to_csv(
        RESULTS_DIR / "kmeans_evaluation.csv",
        index=False,
        encoding="utf-8-sig"
    )
    pd.DataFrame(gmm_results["records"]).to_csv(
        RESULTS_DIR / "gmm_evaluation.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(f"Đã lưu dữ liệu phân cụm tại {RESULTS_DIR}.")

    # 6. Tạo biểu đồ kiểm tra và diễn giải.
    print("\n[6] TẠO BIỂU ĐỒ")
    plot_elbow(kmeans_results, PLOTS_DIR, KMEANS_K)
    plot_silhouette(kmeans_results, PLOTS_DIR, KMEANS_K)
    plot_gmm_information_criteria(gmm_results, PLOTS_DIR, gmm_config)
    plot_pca_cluster_scatter(
        X_pca, kmeans_labels, gmm_labels, PLOTS_DIR
    )
    plot_pc1_pc3(X_pca_all, kmeans_labels, gmm_labels, PLOTS_DIR)
    plot_cluster_sizes(kmeans_labels, gmm_labels, PLOTS_DIR)
    plot_cluster_heatmaps(X_scaled, kmeans_labels, gmm_labels, PLOTS_DIR)
    plot_umap_if_available(
        X_scaled, kmeans_labels, gmm_labels, PLOTS_DIR
    )

    # 7. Tóm tắt các đặc điểm nổi bật của các nhóm.
    print("\n[7] TÓM TẮT")
    print("Biến được giữ:", ", ".join(feature_names))
    print(
        f"Không gian phân cụm: PC1-PC{CLUSTERING_PCA_COMPONENTS}, "
        f"giữ {retained_variance:.2%} phương sai; các thành phần còn lại "
        "không tham gia tính khoảng cách cụm."
    )
    print(
        f"K-Means: K={KMEANS_K}; "
        f"GMM: K={GMM_K}, "
        f"covariance_type={gmm_config['covariance_type']}."
    )
    print(
        "Diễn giải: phân khúc mô tả trên phổ hành vi liên tục; "
        "không có bằng chứng về ranh giới tách biệt hoàn toàn."
    )
    print_cluster_insights(X_scaled, kmeans_labels, gmm_labels)
    print(f"\nCSV so sánh: {RESULTS_DIR / 'model_comparison.csv'}")
    print(f"Biểu đồ: {PLOTS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
