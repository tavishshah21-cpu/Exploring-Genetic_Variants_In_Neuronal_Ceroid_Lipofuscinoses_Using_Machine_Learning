# ==========================================
# BUILD CLN3-SPECIFIC ML DATASET (RUN ONCE)
# ==========================================

import pandas as pd

GENE = "CLN3"

VCF_FILE = "D:/tavish_shah/PES/capstone/Pipeline_1_Statistics/clinvar_20260415.vcf"
ANNOVAR_FILE = "D:/tavish_shah/PES/capstone/Pipeline_1_Statistics/final_merged.csv"

print("🚀 Extracting labels from VCF...")

# ==========================================
# EXTRACT LABELS FROM VCF
# ==========================================
labels = []

with open(VCF_FILE) as f:
    for line in f:
        if line.startswith("#"):
            continue

        parts = line.strip().split("\t")
        chrom = parts[0]
        pos = parts[1]
        info = parts[7]

        clnsig = None
        gene = None

        for field in info.split(";"):
            if field.startswith("CLNSIG="):
                clnsig = field.replace("CLNSIG=", "")
            if field.startswith("GENEINFO="):
                gene = field.replace("GENEINFO=", "").split(":")[0]

        labels.append([chrom, int(pos), clnsig, gene])

labels_df = pd.DataFrame(labels, columns=["Chr", "Start", "CLNSIG", "GeneSymbol"])

print("Total VCF labels:", labels_df.shape)

# ==========================================
# FILTER TO CLN3 ONLY
# ==========================================
labels_df = labels_df[labels_df["GeneSymbol"] == GENE]
print(f"CLN3 variants in VCF: {labels_df.shape[0]}")

# ==========================================
# LOAD ANNOVAR OUTPUT
# ==========================================
print("📊 Loading ANNOVAR data...")
ann = pd.read_csv(ANNOVAR_FILE, low_memory=False)

ann["Start"] = ann["Start"].astype(int)
labels_df["Start"] = labels_df["Start"].astype(int)

ann["Chr"] = ann["Chr"].astype(str).str.replace("chr", "", regex=False)
labels_df["Chr"] = labels_df["Chr"].astype(str).str.replace("chr", "", regex=False)

# ==========================================
# MERGE
# ==========================================
print("🔗 Merging...")
df = ann.merge(labels_df, on=["Chr", "Start"], how="inner")

print("After merge:", df.shape)

# ==========================================
# LABEL MAPPING
# ==========================================
def map_label(x):
    if pd.isna(x):
        return None

    x = str(x).lower()

    if x in ["5", "4"] or "pathogenic" in x:
        return 1
    if x in ["2", "1"] or "benign" in x:
        return 0

    return None

df["label"] = df["CLNSIG"].apply(map_label)
df = df[df["label"].notna()]

print("After labeling:", df.shape)
print("Pathogenic:", (df["label"] == 1).sum())
print("Benign    :", (df["label"] == 0).sum())

# ==========================================
# SAVE CLN3 DATASET
# ==========================================
output_file = "D:/tavish_shah/PES/capstone/Pipeline_1_Statistics/CLN3_ML_READY.csv"
df.to_csv(output_file, index=False)

print("🎉 Saved:", output_file)