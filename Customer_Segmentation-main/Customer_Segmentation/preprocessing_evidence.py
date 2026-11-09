
from pathlib import Path
import pandas as pd
import numpy as np

from data_preprocessing import load_and_preprocess

# 1. Cấu Hình

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "CC GENERAL.csv"

OUTPUT_DIR = (
    BASE_DIR / "results" / "preprocessing_evidence"
)
# 2. Hàm Thống Kê Dữ Liệu
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

# 3. Hàm MAIN 


def main():

    print("\n======================================")
    print("   MINH CHUNG QUY TRINH TIEN XU LY")
    print("======================================")

    # Tạo thư mục lưu kết quả
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    # 1. ĐỌC DỮ LIỆU GỐC

    print("\n[1] Dang doc du lieu goc...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay file du lieu: {DATA_PATH}"
        )

    df_before = pd.read_csv(DATA_PATH)

    print("So dong:", len(df_before))
    print("So cot:", len(df_before.columns))

    # 2. LƯU DỮ LIỆU TRƯỚC TIỀN XỬ LÝ

    print("\n[2] Dang luu du lieu truoc tien xu ly...")

    df_before.head(10).to_csv(
        OUTPUT_DIR / "01_du_lieu_truoc_tien_xu_ly.csv",
        index=False
    )

    # 3. THỐNG KÊ DỮ LIỆU TRƯỚC TIỀN XỬ LÝ

    print("\n[3] Dang thong ke du lieu truoc tien xu ly...")

    report_before = data_report(df_before)

    report_before.to_csv(
        OUTPUT_DIR / "02_thong_ke_truoc_tien_xu_ly.csv",
        encoding="utf-8-sig"
    )

    # 4. THỰC HIỆN TIỀN XỬ LÝ

    print("\n[4] Dang thuc hien tien xu ly du lieu...")

    df_clean, X_scaled, X_pca, feature_names = (
        load_and_preprocess(str(DATA_PATH))
    )

    print("So dong sau lam sach:", len(df_clean))
    print("So thuoc tinh sau tien xu ly:", len(feature_names))

    # 5. LƯU DỮ LIỆU SAU LÀM SẠCH

    print("\n[5] Dang luu du lieu sau lam sach...")

    df_clean.head(10).to_csv(
        OUTPUT_DIR / "03_du_lieu_sau_lam_sach.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 6. LƯU DỮ LIỆU SAU CHUẨN HÓA

    print("\n[6] Dang luu du lieu sau chuan hoa...")

    scaled_df = pd.DataFrame(
        X_scaled,
        columns=feature_names,
        index=df_clean.index
    )

    scaled_df.head(10).to_csv(
        OUTPUT_DIR / "04_du_lieu_sau_chuan_hoa.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 7. THỐNG KÊ VÀ SO SÁNH TRƯỚC - SAU

    print("\n[7] Dang so sanh du lieu truoc va sau...")

    summary = pd.DataFrame([
        {
            "giai_doan": "Truoc tien xu ly",
            "so_dong": len(df_before),
            "so_cot": len(df_before.columns),
            "so_gia_tri_thieu": int(
                df_before.isna().sum().sum()
            ),
            "so_dong_trung_lap": int(
                df_before.duplicated().sum()
            )
        },
        {
            "giai_doan": "Sau lam sach",
            "so_dong": len(df_clean),
            "so_cot": len(df_clean.columns),
            "so_gia_tri_thieu": int(
                df_clean.isna().sum().sum()
            ),
            "so_dong_trung_lap": int(
                df_clean.duplicated().sum()
            )
        }
    ])

    summary.to_csv(
        OUTPUT_DIR / "05_so_sanh_truoc_sau.csv",
        index=False,
        encoding="utf-8-sig"
    )


    # -------------------------------------------------
    # 8. HIỂN THỊ KẾT QUẢ
    # -------------------------------------------------

    print("\n======================================")
    print("       MINH CHUNG TIEN XU LY")
    print("======================================")

    print(summary.to_string(index=False))

    print("\nDa luu minh chung tai:")
    print(OUTPUT_DIR)

    print("\n======================================")
    print("      HOAN THANH TIEN XU LY")
    print("======================================")


# =====================================================
# 4. PROGRAM ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
