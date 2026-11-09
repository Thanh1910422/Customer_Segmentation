import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(file_path, include_additional_features=False):
    """Đọc, làm sạch, chọn biến và tạo dữ liệu chuẩn hóa/PCA."""
    df_original = pd.read_csv(file_path)
    print("Kích thước dữ liệu ban đầu:", df_original.shape)

    df_processed = df_original.copy()
    feature_columns = [
        column for column in df_processed.select_dtypes(include=np.number)
        if column != "CUST_ID"
    ]
    if not feature_columns:
        raise ValueError("Không tìm thấy biến số để phân cụm.")

    features = df_processed[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    empty_columns = features.columns[features.isna().all()].tolist()
    if empty_columns:
        features = features.drop(columns=empty_columns)
        feature_columns = features.columns.tolist()
        print("Loại biến toàn giá trị thiếu:", ", ".join(empty_columns))
    if features.empty or not feature_columns:
        raise ValueError("Không còn biến hợp lệ sau khi kiểm tra dữ liệu.")

    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(
        imputer.fit_transform(features),
        columns=feature_columns,
        index=df_processed.index
    )
    for column in feature_columns:
        df_processed[column] = imputed[column]

    suggested_features = [
        "BALANCE",
        "PURCHASES",
        "CASH_ADVANCE",
        "CREDIT_LIMIT",
        "PAYMENTS",
        "PURCHASES_FREQUENCY",
        "CASH_ADVANCE_FREQUENCY",
        "PRC_FULL_PAYMENT"
    ]
    additional_features = [
        "PURCHASES_TRX",
        "MINIMUM_PAYMENTS",
        "TENURE"
    ]
    selected_candidates = [
        column for column in suggested_features if column in feature_columns
    ]
    if include_additional_features:
        selected_candidates.extend(
            column for column in additional_features
            if column in feature_columns
        )
    if not selected_candidates:
        raise ValueError("Không tìm thấy các biến trong shortlist gợi ý.")
    excluded = [column for column in feature_columns
                if column not in selected_candidates]
    print("Tập biến đầu vào:", ", ".join(selected_candidates))
    if excluded:
        print(
            "Biến ngoài shortlist, không dùng để phân cụm:",
            ", ".join(excluded)
        )

    # Loại từng biến có tương quan cao nhất; giữ biến ít tương quan hơn với
    # các biến còn lại, và dùng thứ tự cột ban đầu để phá hòa ổn định.
    remaining = selected_candidates.copy()
    dropped_features = []
    while len(remaining) > 1:
        corr = imputed[remaining].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.stack().sort_values(ascending=False)
        if pairs.empty or pairs.iloc[0] <= 0.8:
            break
        left, right = pairs.index[0]
        left_mean = corr[left].drop(left).mean()
        right_mean = corr[right].drop(right).mean()
        dropped = right if right_mean >= left_mean else left
        kept = left if dropped == right else right
        dropped_features.append((dropped, kept, float(corr.loc[left, right])))
        remaining.remove(dropped)

    if dropped_features:
        print("Biến bị loại do tương quan tuyệt đối > 0.8:")
        for dropped, kept, value in dropped_features:
            print(f"  - {dropped}: tương quan {value:.3f} với {kept}")
    else:
        print("Không có biến nào vượt ngưỡng tương quan 0.8.")

    # Winsorize, rồi chỉ log1p các biến lệch phải.
    model_features = imputed[remaining].copy()
    for column in remaining:
        low, high = model_features[column].quantile([0.01, 0.99])
        model_features[column] = model_features[column].clip(low, high)

    log_candidates = [
        "BALANCE",
        "PURCHASES",
        "CASH_ADVANCE",
        "CREDIT_LIMIT",
        "PAYMENTS",
        "MINIMUM_PAYMENTS",
        "PURCHASES_TRX"
    ]

    log_columns = []

    for column in remaining:
        if column in log_candidates:
            if model_features[column].min() < 0:
                raise ValueError(
                    f"{column} có giá trị âm, không thể Log1p trực tiếp."
                )

            if model_features[column].skew() > 1:
                model_features[column] = np.log1p(
                    model_features[column]
                )
                log_columns.append(column)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(model_features),
        columns=remaining,
        index=df_processed.index
    )

    pca = PCA(n_components=0.95, svd_solver="full")
    pca_values = pca.fit_transform(X_scaled)
    pca_columns = [f"PC{i + 1}" for i in range(pca_values.shape[1])]
    X_pca = pd.DataFrame(
        pca_values, columns=pca_columns, index=df_processed.index
    )
    X_pca.attrs["explained_variance_ratio_"] = (
        pca.explained_variance_ratio_.copy()
    )
    X_pca.attrs["n_components"] = pca.n_components_

    print(f"Số biến sau chọn lọc: {len(remaining)}")
    print("Biến được giữ:", ", ".join(remaining))
    print("Biến log1p (skew > 1):", ", ".join(log_columns) or "không có")
    print(
        f"PCA giữ {pca.n_components_} thành phần, "
        f"giải thích {pca.explained_variance_ratio_.sum():.2%} phương sai."
    )
    print("Số dòng được giữ nguyên:", len(df_processed))

    return df_processed, X_scaled, X_pca, remaining
