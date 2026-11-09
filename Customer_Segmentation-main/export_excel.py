
from pathlib import Path
import re
import pandas as pd

from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


# ==========================================
# 1. PATH
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "CUSTOMER_SEGMENTATION_REPORT.xlsx"


# ==========================================
# 2. READ CSV
# ==========================================

def read_csv_file(path):

    for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                sep=None,
                engine="python"
            )
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Cannot read: {path}")


# ==========================================
# 3. FIND CSV FILES
# ==========================================

csv_files = sorted([
    p for p in ROOT_DIR.rglob("*.csv")
    if p.is_file()
    and OUTPUT_DIR not in p.parents
    and ".venv" not in p.parts
    and "venv" not in p.parts
])

print("CSV files found:", len(csv_files))


# ==========================================
# 4. EXPORT TO EXCEL
# ==========================================

used_names = set()
overview = []

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    for path in csv_files:

        try:
            df = read_csv_file(path)

            # Clean column names
            df.columns = [
                str(c).strip() if str(c).strip()
                else f"Column_{i+1}"
                for i, c in enumerate(df.columns)
            ]

            # Unique sheet name
            name = re.sub(
                r'[\[\]:*?/\\]',
                '_',
                path.stem
            )[:31]

            base = name or "Sheet"
            counter = 1

            while name.lower() in used_names:
                suffix = f"_{counter}"
                name = base[:31-len(suffix)] + suffix
                counter += 1

            used_names.add(name.lower())

            df.to_excel(
                writer,
                sheet_name=name,
                index=False
            )

            overview.append({
                "Sheet": name,
                "CSV file": str(path.relative_to(ROOT_DIR)),
                "Rows": len(df),
                "Columns": len(df.columns)
            })

            print("[OK]", path.name)

        except Exception as e:
            print("[ERROR]", path, e)

    # Overview sheet
    pd.DataFrame(overview).to_excel(
        writer,
        sheet_name="Tong_quan",
        index=False
    )


# ==========================================
# 5. FORMAT WORKBOOK
# ==========================================

wb = load_workbook(OUTPUT_FILE)

for ws in wb.worksheets:

    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "A2"
    ws.sheet_properties.pageSetUpPr.fitToPage = True

    ws.row_dimensions[1].height = 30

    # Header
    for cell in ws[1]:

        cell.fill = PatternFill(
            fill_type="solid",
            fgColor="1F4E78"
        )

        cell.font = Font(
            name="Arial",
            size=11,
            bold=True,
            color="FFFFFF"
        )

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True
        )

    # Body
    for row in ws.iter_rows(min_row=2):

        for cell in row:

            cell.font = Font(
                name="Arial",
                size=10,
                color="333333"
            )

            cell.alignment = Alignment(
                vertical="center"
            )

    # Column widths
    for col in ws.columns:

        letter = get_column_letter(col[0].column)

        max_length = 0

        for cell in col[:3000]:

            if cell.value is not None:
                max_length = max(
                    max_length,
                    len(str(cell.value))
                )

        ws.column_dimensions[letter].width = min(
            max(max_length + 3, 13),
            45
        )

    # Filter
    if ws.max_row >= 1 and ws.max_column >= 1:
        ws.auto_filter.ref = ws.dimensions


# ==========================================
# 6. SAVE
# ==========================================

wb.save(OUTPUT_FILE)

# Verify file can be reopened
check = load_workbook(
    OUTPUT_FILE,
    read_only=True
)

print("\nEXCEL CREATED SUCCESSFULLY!")
print("Sheets:", check.sheetnames)
print("Output:", OUTPUT_FILE)

check.close()
