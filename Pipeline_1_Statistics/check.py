import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics"

for year in [2025, 2026]:

    print(f"\n{'='*60}")
    print(f"Processing variant_summary_{year}.txt")
    print(f"{'='*60}")

    # --------------------------------------------------
    # LOAD
    # --------------------------------------------------
    df = pd.read_csv(
        os.path.join(BASE_DIR, f"variant_summary_{year}.txt"),
        sep="\t",
        low_memory=False
    )
    df.columns = df.columns.str.strip().str.replace("#", "")

    print(f"Total rows in variant summary : {len(df):,}")

    # --------------------------------------------------
    # FILTER TO CLN3 + GRCh38
    # --------------------------------------------------
    df = df[df["GeneSymbol"] == "CLN3"]
    print(f"After CLN3 filter             : {len(df):,}")

    df = df[df["Assembly"] == "GRCh38"]
    print(f"After GRCh38 filter           : {len(df):,}")

    # --------------------------------------------------
    # SAVE TO CSV
    # --------------------------------------------------
    out_file = os.path.join(BASE_DIR, f"CLN3_{year}_GRCh38.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

    # --------------------------------------------------
    # READ BACK AND PRINT INFO
    # --------------------------------------------------
    df = pd.read_csv(out_file, low_memory=False)

    print(f"\nShape  : {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # --------------------------------------------------
    # CLASSIFY INTO P / B / VUS
    # --------------------------------------------------
    def classify(sig):
        sig = str(sig).lower()
        if "pathogenic" in sig and "benign" not in sig:
            return "Pathogenic"
        elif "benign" in sig and "pathogenic" not in sig:
            return "Benign"
        elif "uncertain" in sig or "vus" in sig:
            return "VUS"
        elif "conflicting" in sig:
            return "Conflicting"
        else:
            return "Other"

    df["Category"] = df["ClinicalSignificance"].apply(classify)

    print(f"\nClinical significance distribution:")
    print(df["Category"].value_counts())

    # --------------------------------------------------
    # PLOT DISTRIBUTION
    # --------------------------------------------------
    counts = df["Category"].value_counts()

    colors = {
        "Pathogenic":   "#D85A30",
        "Benign":       "#1D9E75",
        "VUS":          "#378ADD",
        "Conflicting":  "#BA7517",
        "Other":        "#888780"
    }

    bar_colors = [colors.get(cat, "#888780") for cat in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    axes[0].bar(counts.index, counts.values, color=bar_colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title(f"CLN3 Variant Distribution {year} (GRCh38)", fontsize=13)
    axes[0].set_xlabel("Clinical Significance Category")
    axes[0].set_ylabel("Number of Variants")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontsize=10)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=counts.index,
        colors=bar_colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 0.8}
    )
    axes[1].set_title(f"CLN3 Variant Proportion {year} (GRCh38)", fontsize=13)

    plt.tight_layout()

    plot_file = os.path.join(BASE_DIR, f"CLN3_{year}_distribution.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: CLN3_{year}_distribution.png")