
from pathlib import Path
import pandas as pd
import numpy as np

from data_preprocessing import load_and_preprocess

# Thư mục dự án
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "CC GENERAL.csv"
OUTPUT_DIR = BASE_DIR / "results" / "preprocessing_evidence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Đọc dữ liệu gốc
df_before = pd.read_csv(DATA_PATH)

# 2. Lưu mẫu dữ liệu trước tiền xử lý
df_before.head(10).to_csv(
    OUTPUT_DIR / "01_du_lieu_truoc_tien_xu_ly.csv",
    index=False
)

# 3. Thống kê dữ liệu trước tiền xử lý
def data_report(df):
    """Tạo thống kê chất lượng dữ liệu theo từng cột."""
    return pd.DataFrame({
        "so_gia_tri_thieu": df.isna().sum(),
        "so_gia_tri_duy_nhat": df.nunique(dropna=True),
        "kieu_du_lieu": df.dtypes.astype(str),
        "so_gia_tri_vo_cuc": df.select_dtypes(
            include=np.number
        ).apply(lambda col: np.isinf(col).sum())
    })

data_report(df_before).to_csv(
    OUTPUT_DIR / "02_thong_ke_truoc_tien_xu_ly.csv"
)

# 4. Chạy đúng quy trình tiền xử lý của dự án
df_clean, X_scaled, X_pca, feature_names = load_and_preprocess(str(DATA_PATH))

# 5. Lưu dữ liệu sau làm sạch, trước chuẩn hóa
df_clean.head(10).to_csv(
    OUTPUT_DIR / "03_du_lieu_sau_lam_sach.csv",
    index=False
)

# 6. Lưu dữ liệu sau chuẩn hóa
scaled_df = pd.DataFrame(
    X_scaled,
    columns=feature_names,
    index=df_clean.index
)

scaled_df.head(10).to_csv(
    OUTPUT_DIR / "04_du_lieu_sau_chuan_hoa.csv",
    index=False
)

# 7. Thống kê tổng quan trước và sau
summary = pd.DataFrame([
    {
        "giai_doan": "Truoc tien xu ly",
        "so_dong": len(df_before),
        "so_cot": len(df_before.columns),
        "so_gia_tri_thieu": int(df_before.isna().sum().sum()),
        "so_dong_trung_lap": int(df_before.duplicated().sum())
    },
    {
  
      "giai_doan": "Sau lam sach",
        "so_dong": len(df_clean),
        "so_cot": len(df_clean.columns),
        "so_gia_tri_thieu": int(df_clean.isna().sum().sum()),
        "so_dong_trung_lap": int(df_clean.duplicated().sum())
    }
])

summary.to_csv(
    OUTPUT_DIR / "05_so_sanh_truoc_sau.csv",
    index=False
)

print("\n=== MINH CHUNG TIEN XU LY ===")
print(summary.to_string(index=False))
print(f"\nDa luu minh chung tai: {OUTPUT_DIR}")
