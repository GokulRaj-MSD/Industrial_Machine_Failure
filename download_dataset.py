"""Download the UCI AI4I 2020 Predictive Maintenance Dataset."""
from pathlib import Path
from ucimlrepo import fetch_ucirepo

OUT = Path("data/ai4i2020.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

dataset = fetch_ucirepo(id=601)
df = dataset.data.original
# Save the original UCI table so the project remains reproducible.
df.to_csv(OUT, index=False)
print(f"Saved {len(df):,} rows to {OUT}")
print("Columns:")
for c in df.columns:
    print(" -", c)
