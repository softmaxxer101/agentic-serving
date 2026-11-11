import sys
import pandas as pd

m = pd.read_csv(sys.argv[1])
pattern = sys.argv[2] if len(sys.argv) > 2 else "prefix_cache"

d = m[m["name"].str.contains(pattern, na=False)].copy()
d["value"] = d["value"].astype(float)

for (name, labels), group in d.groupby(["name", "labels"], dropna=False):
    print(
        name,
        labels,
        "delta",
        group["value"].iloc[-1] - group["value"].iloc[0],
    )
