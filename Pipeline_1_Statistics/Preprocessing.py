# ==========================================
# CLN3-SPECIFIC PREPROCESSING PIPELINE
# With AlphaMissense + Structural Features
# + CADD-Indel for insertions/deletions
# GRCh38 only
# ==========================================

import re
import subprocess
import pandas as pd

WSL_ANNOVAR_DIR  = "~/annovar_work/annovar"
CADD_INDEL_FILE  = "~/cadd_data/gnomad.genomes.r4.0.indel.tsv.gz"

GENE = "CLN3"

AM_FILE  = r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics\AlphaMissense-Search-Q13286.tsv"
PDB_FILE = r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics\cln3_structural_features.csv"

THREE_TO_ONE = {
    'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C',
    'Gln':'Q','Glu':'E','Gly':'G','His':'H','Ile':'I',
    'Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P',
    'Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V'
}


def extract_position_and_alt(aa_change):
    # Single letter: p.N981K
    match = re.search(r'p\.([A-Z])(\d+)([A-Z])', str(aa_change))
    if match:
        return int(match.group(2)), match.group(3)
    # Three letter: p.Asn981Lys
    match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', str(aa_change))
    if match:
        return int(match.group(2)), THREE_TO_ONE.get(match.group(3), None)
    return None, None


def load_alphamissense():
    print("\n📂 Loading AlphaMissense data...")
    am = pd.read_csv(AM_FILE, sep="\t")
    am = am[["position", "a.a.2", "pathogenicity score", "pathogenicity class"]]
    am.columns = ["residue_position", "alt_aa", "am_pathogenicity", "am_class"]
    am["residue_position"] = pd.to_numeric(am["residue_position"], errors="coerce")
    print(f"AlphaMissense entries: {am.shape[0]}")
    return am


def load_structural_features():
    print("\n📂 Loading structural features...")
    pdb = pd.read_csv(PDB_FILE)
    pdb = pdb[["residue_number", "pLDDT", "secondary_structure", "RSA", "burial"]]
    pdb.columns = ["residue_position", "pLDDT", "secondary_structure", "RSA", "burial"]
    pdb["residue_position"] = pd.to_numeric(pdb["residue_position"], errors="coerce")
    print(f"Structural feature entries: {pdb.shape[0]}")
    return pdb


def query_cadd_indel(chrom, pos, ref, alt):
    """Query CADD-Indel file for a single variant using tabix via WSL"""
    try:
        region = f"{chrom}:{pos}-{pos}"
        cmd = [
            "wsl", "bash", "-c",
            f"tabix {CADD_INDEL_FILE} {region}"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 6:
                if parts[2] == ref and parts[3] == alt:
                    return float(parts[4]), float(parts[5])
    except Exception:
        pass
    return None, None


def get_cadd_indel_scores(df_indels):
    """Get CADD-Indel scores for a dataframe of indel variants"""
    print(f"\n🔍 Querying CADD-Indel for {len(df_indels)} indel variants...")

    cadd_raw_scores  = []
    cadd_phred_scores = []

    for i, (_, row) in enumerate(df_indels.iterrows()):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(df_indels)}")

        chrom = str(row["Chromosome"]).replace("chr", "")
        pos   = int(row["Start"])
        ref   = str(row["ReferenceAllele"])
        alt   = str(row["AlternateAllele"])

        raw, phred = query_cadd_indel(chrom, pos, ref, alt)
        cadd_raw_scores.append(raw)
        cadd_phred_scores.append(phred)

    df_indels = df_indels.copy()
    df_indels["CADD_raw"]   = cadd_raw_scores
    df_indels["CADD_phred"] = cadd_phred_scores

    matched = sum(1 for x in cadd_phred_scores if x is not None)
    print(f"CADD-Indel matched: {matched} / {len(df_indels)}")

    return df_indels


def run_pipeline(INPUT_FILE, OUTPUT_PREFIX):
    print(f"\n🚀 Processing: {INPUT_FILE}")

    am  = load_alphamissense()
    pdb = load_structural_features()

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------
    df = pd.read_csv(INPUT_FILE, sep="\t", low_memory=False)
    df.columns = df.columns.str.strip().str.replace("#", "")

    df = df[[
        "VariationID", "GeneSymbol", "ClinicalSignificance",
        "Chromosome", "PositionVCF", "Assembly",
        "ReferenceAlleleVCF", "AlternateAlleleVCF"
    ]]

    df = df.rename(columns={
        "ReferenceAlleleVCF": "ReferenceAllele",
        "AlternateAlleleVCF": "AlternateAllele",
        "PositionVCF":        "Start"
    })

    # --------------------------------------------------
    # FILTER TO CLN3 + GRCh38
    # --------------------------------------------------
    df = df[df["GeneSymbol"] == GENE]
    print(f"CLN3 variants found: {df.shape[0]}")

    df = df[df["Assembly"] == "GRCh38"]
    print(f"After GRCh38 filter: {df.shape[0]}")

    if df.shape[0] == 0:
        print("❌ No GRCh38 CLN3 variants found.")
        return

    # --------------------------------------------------
    # LABEL CREATION
    # --------------------------------------------------
    df["ClinicalSignificance"] = df["ClinicalSignificance"].astype(str).str.lower()
    valid = ["pathogenic", "likely pathogenic", "benign", "likely benign"]
    df = df[df["ClinicalSignificance"].isin(valid)]

    df["label"] = df["ClinicalSignificance"].map({
        "pathogenic":        1,
        "likely pathogenic": 1,
        "benign":            0,
        "likely benign":     0
    })

    df = df.drop_duplicates(subset="VariationID")

    print("After label filter:", df.shape[0])
    print("Pathogenic:", (df["label"] == 1).sum())
    print("Benign    :", (df["label"] == 0).sum())

    # --------------------------------------------------
    # REMOVE INVALID VARIANTS
    # --------------------------------------------------
    df["ReferenceAllele"] = df["ReferenceAllele"].astype(str).str.strip()
    df["AlternateAllele"] = df["AlternateAllele"].astype(str).str.strip()

    df = df[
        df["ReferenceAllele"].notna() &
        df["AlternateAllele"].notna() &
        (df["ReferenceAllele"].str.lower() != "na") &
        (df["AlternateAllele"].str.lower() != "na") &
        (df["ReferenceAllele"] != ".") &
        (df["AlternateAllele"] != ".") &
        (df["ReferenceAllele"] != "") &
        (df["AlternateAllele"] != "") &
        df["Chromosome"].notna() &
        (df["Start"] > 0)
    ]

    df = df[~df["Chromosome"].astype(str).str.contains("na", case=False)]

    print("After removing invalid:", df.shape[0])

    # --------------------------------------------------
    # SPLIT INTO SNPs AND INDELS
    # --------------------------------------------------
    is_snp = (
        (df["ReferenceAllele"].str.len() == 1) &
        (df["AlternateAllele"].str.len() == 1)
    )

    df_snps   = df[is_snp].copy()
    df_indels = df[~is_snp].copy()

    print(f"\nSNPs   : {len(df_snps)}")
    print(f"Indels : {len(df_indels)}")

    # --------------------------------------------------
    # PROCESS SNPs — ANNOVAR
    # --------------------------------------------------
    print("\n--- Processing SNPs via ANNOVAR ---")

    df_snps["Chromosome"] = "chr" + df_snps["Chromosome"].astype(str)

    annovar_input = OUTPUT_PREFIX + ".avinput"
    annovar_df = df_snps[["Chromosome", "Start", "Start",
                           "ReferenceAllele", "AlternateAllele"]].copy()
    annovar_df.columns = ["chr", "start", "end", "ref", "alt"]
    annovar_df.to_csv(annovar_input, sep="\t", header=False, index=False)

    print("✅ ANNOVAR input created:", annovar_input)

    wsl_input  = annovar_input.replace("D:", "/mnt/d").replace("\\", "/")
    wsl_output = OUTPUT_PREFIX.replace("D:", "/mnt/d").replace("\\", "/")

    cmd = [
        "wsl", "bash", "-c",
        f"cd {WSL_ANNOVAR_DIR} && "
        f"./table_annovar.pl {wsl_input} humandb/ "
        f"-buildver hg38 "
        f"-out {wsl_output} "
        f"-remove "
        f"-protocol refGene,avsnp150,dbnsfp42a "
        f"-operation g,f,f "
        f"-nastring . "
        f"-csvout "
        f"-otherinfo "
        f"-arg '-splicing 5',,"
    ]

    print("⚙️ Running ANNOVAR...")
    subprocess.run(cmd, check=True)

    ann_file = OUTPUT_PREFIX + ".hg38_multianno.csv"
    df_ann = pd.read_csv(ann_file)

    print("Annotated SNP rows:", df_ann.shape[0])

    df_snps["Start"] = df_snps["Start"].astype(int)
    df_ann["Start"]  = df_ann["Start"].astype(int)

    df_snps = df_snps.merge(
        df_ann,
        left_on=["Chromosome", "Start", "ReferenceAllele", "AlternateAllele"],
        right_on=["Chr", "Start", "Ref", "Alt"],
        how="left"
    )

    print("SNPs after ANNOVAR merge:", df_snps.shape[0])

    # --------------------------------------------------
    # PROCESS INDELS — CADD-INDEL
    # --------------------------------------------------
    print("\n--- Processing Indels via CADD-Indel ---")

    df_indels["Chromosome"] = "chr" + df_indels["Chromosome"].astype(str)
    df_indels = get_cadd_indel_scores(df_indels)

    # --------------------------------------------------
    # COMBINE SNPs + INDELS
    # --------------------------------------------------
    print("\n🔗 Combining SNPs and Indels...")
    df = pd.concat([df_snps, df_indels], ignore_index=True)
    print("Combined dataset:", df.shape[0])

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------
    df["var_length"] = (
        df["AlternateAllele"].str.len() -
        df["ReferenceAllele"].str.len()
    )

    # --------------------------------------------------
    # EXTRACT AMINO ACID POSITION
    # --------------------------------------------------
    print("\n🔗 Extracting amino acid positions...")

    if "AAChange.refGene" in df.columns:
        df["residue_position"] = df["AAChange.refGene"].apply(
            lambda x: extract_position_and_alt(x)[0]
        )
        df["alt_aa"] = df["AAChange.refGene"].apply(
            lambda x: extract_position_and_alt(x)[1]
        )
        df["residue_position"] = pd.to_numeric(df["residue_position"], errors="coerce")
        print("Variants with position extracted:",
              df["residue_position"].notna().sum())
    else:
        df["residue_position"] = None
        df["alt_aa"] = None

    # --------------------------------------------------
    # MERGE ALPHAMISSENSE
    # --------------------------------------------------
    print("\n🔗 Merging AlphaMissense...")
    df = df.merge(am, on=["residue_position", "alt_aa"], how="left")
    matched_am = df["am_pathogenicity"].notna().sum()
    print(f"Matched to AlphaMissense: {matched_am} / {len(df)}")

    df["am_likely_pathogenic"] = (df["am_class"] == "likely_pathogenic").astype(int)
    df["am_ambiguous"]         = (df["am_class"] == "ambiguous").astype(int)

    # --------------------------------------------------
    # MERGE STRUCTURAL FEATURES
    # --------------------------------------------------
    print("\n🔗 Merging structural features...")
    df = df.merge(pdb, on="residue_position", how="left")
    matched_pdb = df["pLDDT"].notna().sum()
    print(f"Matched to structural features: {matched_pdb} / {len(df)}")

    df["ss_helix"] = (df["secondary_structure"] == "H").astype(int)
    df["ss_sheet"] = (df["secondary_structure"] == "E").astype(int)
    df["ss_coil"]  = (df["secondary_structure"] == "-").astype(int)

    df["is_buried"]       = (df["burial"] == "buried").astype(int)
    df["is_exposed"]      = (df["burial"] == "exposed").astype(int)
    df["is_intermediate"] = (df["burial"] == "intermediate").astype(int)

    # --------------------------------------------------
    # SELECT FEATURES
    # --------------------------------------------------
    features = [
        # Conservation
        "LINSIGHT",
        "GERP++_RS",
        "phyloP100way_vertebrate",
        "phastCons100way_vertebrate",
        "SiPhy_29way_logOdds",

        # Functional scores (SNPs)
        "CADD_phred",
        "CADD_raw",
        "REVEL_score",
        "SIFT_score",
        "Polyphen2_HDIV_score",
        "Polyphen2_HVAR_score",
        "MutationTaster_score",
        "MetaRNN_score",
        "ClinPred_score",
        "VEST4_score",
        "MetaSVM_score",
        "MetaLR_score",

        # Variant info
        "var_length",

        # AlphaMissense
        "am_pathogenicity",
        "am_likely_pathogenic",
        "am_ambiguous",

        # Structural
        "pLDDT",
        "RSA",
        "ss_helix",
        "ss_sheet",
        "ss_coil",
        "is_buried",
        "is_exposed",
        "is_intermediate"
    ]

    features = [f for f in features if f in df.columns]
    print(f"\nFeatures available: {len(features)}")
    print(features)

    df = df[features + ["label"]]

    # --------------------------------------------------
    # CLEAN MISSING VALUES
    # --------------------------------------------------
    df = df.replace(".", pd.NA)

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(thresh=int(len(features) * 0.5))

    print("\nFinal CLN3 dataset size:", df.shape)
    print("Pathogenic:", (df["label"] == 1).sum())
    print("Benign    :", (df["label"] == 0).sum())

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------
    out_file = OUTPUT_PREFIX + "_CLN3_ML_READY.csv"
    df.to_csv(out_file, index=False)
    print(f"🎉 CLN3 DATASET SAVED: {out_file}")


# --------------------------------------------------
# RUN
# --------------------------------------------------
run_pipeline(
    r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics\variant_summary_2025.txt",
    r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics\CLN3_2025"
)

run_pipeline(
    r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics\variant_summary_2026.txt",
    r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics\CLN3_2026"
)