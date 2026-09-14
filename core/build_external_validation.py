import gzip
from pathlib import Path

import pandas as pd
import requests

from src.features_config import AR_GENES, PSA_GENES, PROLIF_GENES


external_dir = Path("data/external")
symbols = pd.read_csv("outputs/tables/final_genomic_features.csv")["feature"].astype(str).tolist()
source_symbols = list(dict.fromkeys(symbols + list(PSA_GENES) + list(AR_GENES) + list(PROLIF_GENES)))
known_entrez = {
    "DKFZp547O146": "51341",
    "SUGT1P1": "441394",
    "SPATA41": "388182",
    "BMS1P4": "729096",
}
mapping = {}
missing = []

for symbol in source_symbols:
    if symbol in known_entrez:
        mapping[symbol] = known_entrez[symbol]
        continue
    response = requests.get(
        "https://mygene.info/v3/query",
        params={
            "q": symbol,
            "scopes": "symbol",
            "fields": "entrezgene",
            "species": "human",
            "size": 5,
        },
        timeout=20,
    ).json()
    hits = response.get("hits", [])
    entrez = hits[0].get("entrezgene") if hits else None
    if entrez is None:
        missing.append(symbol)
    else:
        mapping[symbol] = str(entrez)

with gzip.open(external_dir / "GSE54460_FPKM.txt.gz", "rt", errors="replace") as handle:
    header = handle.readline().rstrip("\n").split("\t")
    sample_ids = header[2:]
    metadata = [handle.readline().rstrip("\n").split("\t")[2:] for _ in range(14)]
    handle.readline()  # Entrez/Gene Symbol header
    expression = pd.read_csv(handle, sep="\t", header=None, dtype={0: str})

bcr_values = metadata[4]
valid_indices = [
    index
    for index, (sample_id, target) in enumerate(zip(sample_ids, bcr_values))
    if sample_id and target in {"0", "1"}
]

expression = expression.iloc[:, [0] + [index + 2 for index in valid_indices]]
expression = expression.rename(columns={0: "entrez"})
expression = expression[expression["entrez"].notna()]
numeric_expression = expression.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
expression = pd.concat([expression[["entrez"]], numeric_expression], axis=1)
expression = expression.groupby("entrez", as_index=True).mean(numeric_only=True)

selected_symbols = []
selected_entrez = []
for symbol in source_symbols:
    entrez = mapping.get(symbol)
    if entrez in expression.index:
        selected_symbols.append(symbol)
        selected_entrez.append(entrez)

if len(selected_symbols) != len(source_symbols):
    unavailable = [symbol for symbol in source_symbols if symbol not in selected_symbols]
    raise RuntimeError(f"Could not map all selected genes: {unavailable}")

X_external = expression.loc[selected_entrez].T
X_external.columns = selected_symbols
X_external = X_external.reset_index(drop=True)
y_external = pd.Series(
    [int(bcr_values[index]) for index in valid_indices],
    name="BCR",
)

X_external.to_csv(external_dir / "X_external.csv", index=False)
y_external.to_csv(external_dir / "y_external.csv", index=False)

print(f"Created X_external.csv: {X_external.shape}")
print(f"Created y_external.csv: {y_external.shape}, positives={int(y_external.sum())}")
print(f"Mapped genomic source genes: {len(selected_symbols)}/{len(source_symbols)}")
print(f"Missing from MyGene: {missing}")
