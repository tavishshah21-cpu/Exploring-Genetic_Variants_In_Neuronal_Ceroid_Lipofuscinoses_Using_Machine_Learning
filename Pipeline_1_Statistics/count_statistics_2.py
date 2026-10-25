import pandas as pd

df = pd.read_csv(r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics\variant_summary_2025.txt",
                 sep="\t", low_memory=False)

# Filter for CLN3 gene only
df_cln3 = df[df["GeneSymbol"] == "CLN3"]

# Remove duplicates
df_cln3 = df_cln3.drop_duplicates(subset="VariationID")

print(f"Total CLN3 variants (after dedup): {len(df_cln3)}")

print("\n=== Clinical Significance Distribution ===")
print(df_cln3["ClinicalSignificance"].value_counts())

print("\n=== Type Distribution ===")
print(df_cln3["Type"].value_counts())

# Filter to Pathogenic/Benign only
df_cln3["ClinicalSignificance"] = df_cln3["ClinicalSignificance"].astype(str).str.lower()
valid = ["pathogenic", "likely pathogenic", "benign", "likely benign"]
df_cln3_filtered = df_cln3[df_cln3["ClinicalSignificance"].isin(valid)]

print(f"\nCLN3 Pathogenic/Benign only: {len(df_cln3_filtered)}")

# Missense in CLN3
df_cln3_filtered["is_missense"] = df_cln3_filtered["Name"].astype(str).str.contains(
    r"p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}", regex=True)

missense_count = df_cln3_filtered["is_missense"].sum()
total = len(df_cln3_filtered)

print(f"\nCLN3 Missense variants : {missense_count}")
print(f"Percentage             : {missense_count/total*100:.1f}%")

print("\n=== CLN3 Missense by Label ===")
df_cln3_filtered["label"] = df_cln3_filtered["ClinicalSignificance"].map({
    "pathogenic": 1,
    "likely pathogenic": 1,
    "benign": 0,
    "likely benign": 0
})
print(df_cln3_filtered[df_cln3_filtered["is_missense"]]["label"].value_counts().rename({0: "Benign", 1: "Pathogenic"}))