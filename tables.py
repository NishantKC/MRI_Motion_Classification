import re
import pandas as pd
from pathlib import Path

# ---- INPUT ----
path = Path("ad_motion_metrics_results.txt")
txt = path.read_text()

# ---- Helper to parse the fixed-width tables in your .txt ----
def parse_table(block_title: str, txt: str) -> pd.DataFrame:
    """
    Finds a section like:
      <block_title>
      ------
      <header row>
      ------
      <data rows>
      ------
    and returns it as a DataFrame.
    """
    m = re.search(rf"{re.escape(block_title)}.*?\n-+\n(.*?)\n-+\n", txt, flags=re.S)
    if not m:
        raise ValueError(f"Could not find block titled: {block_title}")

    lines = [ln.rstrip() for ln in m.group(1).splitlines() if ln.strip()]
    header = re.split(r"\s{2,}", lines[0].strip())

    rows = []
    for ln in lines[1:]:
        parts = re.split(r"\s{2,}", ln.strip())
        if len(parts) != len(header):  # fallback
            parts = re.split(r"\s+", ln.strip())
        rows.append(parts)

    df = pd.DataFrame(rows, columns=header)
    df = df.rename(columns={df.columns[0]: "Motion Level"})

    # numeric conversion for all but the first column
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace("+", "", regex=False), errors="coerce")
    return df

# ---- Table 2: performance across motion levels ----
df_metrics = parse_table("METRICS TABLE", txt)

# ---- Table 3: degradation from baseline ----
df_deg = parse_table("DEGRADATION FROM BASELINE (M0 - No Motion)", txt)

# ---- Table 1: baseline only (M0) + samples ----
m0_block = re.search(r"M0 \(No Motion\).*?(?=\n\nM1|\nM1|\Z)", txt, flags=re.S)
baseline = {}
samples = None

if m0_block:
    # Metrics
    for k, v in re.findall(r"^\s*([A-Za-z ]+):\s*([0-9.]+)\s*$", m0_block.group(0), flags=re.M):
        baseline[k.strip()] = float(v)
    # Samples
    sm = re.search(r"^\s*Samples:\s*(\d+)\s*$", m0_block.group(0), flags=re.M)
    if sm:
        samples = int(sm.group(1))

df_baseline = pd.DataFrame([{
    "Motion Level": "M0 (No Motion)",
    "Samples": samples,
    "Accuracy": baseline.get("Accuracy"),
    "Precision": baseline.get("Precision"),
    "Recall": baseline.get("Recall"),
    "F1 Score": baseline.get("F1 Score"),
    "AUROC": baseline.get("AUROC"),
}])

# ---- Optional: round for nice display ----
for dfx in (df_baseline, df_metrics, df_deg):
    for col in dfx.columns:
        if col not in ["Motion Level", "Samples"]:
            for dfx in (df_baseline, df_metrics, df_deg):
                numeric_cols = dfx.select_dtypes(include="number").columns
                dfx[numeric_cols] = dfx[numeric_cols].round(4)

# ---- Export outputs ----
out_dir = Path("data")
csv1 = out_dir / "table1_baseline_m0.csv"
csv2 = out_dir / "table2_performance_by_motion.csv"
csv3 = out_dir / "table3_degradation_from_baseline.csv"
xlsx = out_dir / "ad_motion_tables.xlsx"

df_baseline.to_csv(csv1, index=False)
df_metrics.to_csv(csv2, index=False)
df_deg.to_csv(csv3, index=False)

with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
    df_baseline.to_excel(writer, sheet_name="Table1_Baseline_M0", index=False)
    df_metrics.to_excel(writer, sheet_name="Table2_Performance", index=False)
    df_deg.to_excel(writer, sheet_name="Table3_Degradation", index=False)

print("Wrote:")
print(" -", csv1)
print(" -", csv2)
print(" -", csv3)
print(" -", xlsx)