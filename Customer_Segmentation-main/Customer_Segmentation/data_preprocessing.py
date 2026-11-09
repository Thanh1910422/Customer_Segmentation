
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_and_preprocess(file_path, include_additional_features=True):

    # BƯỚC 1: Đọc dữ liệu
    df = pd.read_csv(file_path)
    df_processed = df.copy()

    # BƯỚC 2: Chọn các cột có kiểu dữ liệu số
    features = df_processed.select_dtypes(include=np.number)

    # Không sử dụng mã khách hàng làm đặc trưng
    if "CUST_ID" in features.columns:
        features = features.drop(columns=["CUST_ID"])

    # BƯỚC 3: Xử lý giá trị vô hạn
    features = features.replace([np.inf, -np.inf], np.nan)

    # Xóa các cột bị thiếu toàn bộ dữ liệu
    features = features.dropna(axis=1, how="all")

    # BƯỚC 4: Điền giá trị thiếu bằng trung vị
    imputer = SimpleImputer(strategy="median")
    features = pd.DataFrame(
        imputer.fit_transform(features),
        columns=features.columns
    )

    # BƯỚC 5: Chọn các đặc trưng sử dụng
    suggested_features = [
        "BALANCE",
        "PURCHASES",
        "CASH_ADVANCE",
        "CREDIT_LIMIT",
        "PAYMENTS",
        "MINIMUM_PAYMENTS",
        "PRC_FULL_PAYMENT",
        "TENURE"
    ]

    additional_features = [
        "BALANCE_FREQUENCY",
        "PURCHASES_FREQUENCY",
        "ONEOFF_PURCHASES",
        "INSTALLMENTS_PURCHASES",
        "CASH_ADVANCE_FREQUENCY",
        "PURCHASES_TRX"
    ]

    selected_features = suggested_features.copy()

    if include_additional_features:
        selected_features += additional_features

    # Chỉ giữ những cột thực sự tồn tại trong dữ liệu
    selected_features = [
        col for col in selected_features
        if col in features.columns
    ]

    features = features[selected_features]

    # BƯỚC 6: Loại bỏ các đặc trưng có tương quan cao
    correlation = features.corr().abs()

    upper_triangle = correlation.where(
        np.triu(np.ones(correlation.shape), k=1).astype(bool)
    )

    columns_to_drop = [
        col for col in upper_triangle.columns
        if any(upper_triangle[col] > 0.8)
    ]

    features = features.drop(columns=columns_to_drop)

    # BƯỚC 7: Xử lý giá trị ngoại lai
    for col in features.columns:
        lower = features[col].quantile(0.01)
        upper = features[col].quantile(0.99)

        features[col] = features[col].clip(lower, upper)

    # BƯỚC 8: Biến đổi các cột có phân phối lệch
    for col in features.columns:
        if features[col].skew() > 1:
            min_value = features[col].min()

            if min_value < 0:
                features[col] = features[col] - min_value

            features[col] = np.log1p(features[col])

    # BƯỚC 9: Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index
    )

    # BƯỚC 10: Giảm chiều dữ liệu bằng PCA
    pca = PCA(n_components=0.90, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Chuyển kết quả PCA thành DataFrame
    X_pca = pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
    )
    X_pca.attrs["explained_variance_ratio_"] = (
        pca.explained_variance_ratio_.copy()
    )

    # BƯỚC 11: Trả về kết quả
    return (
        df_processed,
        X_scaled,
        X_pca,
        features.columns.tolist()
    )
