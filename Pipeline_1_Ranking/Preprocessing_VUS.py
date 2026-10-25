
# ============================================================
# CLN3 VUS PREPROCESSING PIPELINE
# ============================================================
#
# PURPOSE
# ============================================================
#
# Both labelled training variants and VUS variants are
# extracted from:
#
#     variant_summary_2026.txt
#
# Both datasets go through the same annotation pipeline:
#
#   - CLN3 filtering
#   - GRCh38 filtering
#   - Duplicate removal
#   - SNP / INDEL separation
#   - ANNOVAR
#   - dbNSFP
#   - CADD-Indel
#   - AlphaMissense
#   - AlphaFold
#   - pLDDT
#   - RSA
#   - Secondary structure
#   - Burial
#   - Contact count
#   - Feature engineering
#   - Missing-value filtering
#
# LABELS:
#   Pathogenic / Likely Pathogenic = 1
#   Benign / Likely Benign         = 0
#   VUS                            = Unlabelled
#
# OUTPUT:
#
#   Pipeline_1_Ranking/
#
#       CLN3_2026_CLN3_ML_READY.csv
#       CLN3_VUS_CLN3_ML_READY.csv
#
# ============================================================


import os
import re
import subprocess

import numpy as np
import pandas as pd
import pydssp

from Bio.PDB import (
    PDBParser,
    ShrakeRupley,
    NeighborSearch
)


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = (
    r"D:\tavish_shah\PES\capstone"
    r"\Pipeline_1_Statistics"
)

RANKING_DIR = (
    r"D:\tavish_shah\PES\capstone"
    r"\Pipeline_1_Ranking"
)

os.makedirs(RANKING_DIR, exist_ok=True)

GENE = "CLN3"


# ============================================================
# CLINVAR SOURCE: 2026
# ============================================================

CLINVAR_FILE = os.path.join(
    SOURCE_DIR,
    "variant_summary_2026.txt"
)


# ============================================================
# ALPHAMISSENSE
# ============================================================

AM_FILE = os.path.join(
    SOURCE_DIR,
    "AlphaMissense-Search-Q13286.tsv"
)


# ============================================================
# ALPHAFOLD PDB
# ============================================================

PDB_FILE = os.path.join(
    SOURCE_DIR,
    "AF-Q13286-F1-model_v6.pdb"
)


# ============================================================
# STRUCTURAL FEATURES OUTPUT
# ============================================================

PDB_OUTPUT_FILE = os.path.join(
    RANKING_DIR,
    "cln3_structural_features.csv"
)


# ============================================================
# ANNOVAR
# ============================================================

WSL_ANNOVAR_DIR = "~/annovar_work/annovar"


# ============================================================
# CADD-INDEL
# ============================================================

CADD_INDEL_FILE = (
    "~/cadd_data/gnomad.genomes.r4.0.indel.tsv.gz"
)


# ============================================================
# CONTACT RADIUS
# ============================================================

CONTACT_RADIUS = 8.0


# ============================================================
# OUTPUT DATASETS
# ============================================================

LABELED_OUTPUT = os.path.join(
    RANKING_DIR,
    "CLN3_2026_CLN3_ML_READY.csv"
)

VUS_OUTPUT = os.path.join(
    RANKING_DIR,
    "CLN3_VUS_CLN3_ML_READY.csv"
)


# ============================================================
# AMINO ACID CONVERSION
# ============================================================

THREE_TO_ONE = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V"
}


# ============================================================
# ALPHAFOLD STRUCTURAL FEATURE EXTRACTION
# ============================================================

def run_structural_feature_extraction():

    print()
    print("=" * 75)
    print("CLN3 ALPHAFOLD STRUCTURAL FEATURE EXTRACTION")
    print("=" * 75)

    if not os.path.exists(PDB_FILE):
        raise FileNotFoundError(
            "\nAlphaFold PDB not found:\n" + PDB_FILE
        )

    print()
    print("Loading AlphaFold structure...")

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure(
        "CLN3",
        PDB_FILE
    )

    model = structure[0]

    # --------------------------------------------------------
    # pLDDT
    # --------------------------------------------------------

    plddt_data = []

    for chain in model:
        for residue in chain:

            residue_number = residue.get_id()[1]
            amino_acid = residue.get_resname()

            if "CA" in residue:

                plddt = residue["CA"].get_bfactor()

                plddt_data.append({
                    "residue_number": residue_number,
                    "amino_acid": amino_acid,
                    "pLDDT": plddt
                })

    df_plddt = pd.DataFrame(plddt_data)

    print("Residues extracted:", len(df_plddt))

    # --------------------------------------------------------
    # SECONDARY STRUCTURE
    # --------------------------------------------------------

    print()
    print("Calculating secondary structure...")

    with open(PDB_FILE, "r") as f:
        pdb_text = f.read()

    coords = pydssp.read_pdbtext(pdb_text)

    ss_results = pydssp.assign(
        coords,
        out_type="c3"
    )

    # Assign the first chain's residue-level secondary
    # structure only when the returned length matches.
    if len(ss_results) != len(df_plddt):
        raise ValueError(
            "Secondary-structure output length does not match "
            "the extracted AlphaFold residues. Check the PDB "
            "chain and residue handling."
        )

    df_plddt["secondary_structure"] = list(ss_results)

    # --------------------------------------------------------
    # RSA
    # --------------------------------------------------------

    print()
    print("Calculating RSA...")

    parser2 = PDBParser(QUIET=True)

    structure2 = parser2.get_structure(
        "CLN3",
        PDB_FILE
    )

    sr = ShrakeRupley()

    sr.compute(
        structure2,
        level="R"
    )

    max_asa = {
        "ALA": 121,
        "ARG": 265,
        "ASN": 187,
        "ASP": 187,
        "CYS": 148,
        "GLN": 214,
        "GLU": 214,
        "GLY": 97,
        "HIS": 216,
        "ILE": 195,
        "LEU": 191,
        "LYS": 230,
        "MET": 203,
        "PHE": 228,
        "PRO": 154,
        "SER": 143,
        "THR": 163,
        "TRP": 264,
        "TYR": 255,
        "VAL": 165
    }

    rsa_data = []

    for chain in structure2[0]:
        for residue in chain:

            residue_number = residue.get_id()[1]
            residue_name = residue.get_resname()

            asa = residue.sasa

            max_value = max_asa.get(
                residue_name,
                200
            )

            rsa = round(asa / max_value, 3)

            rsa_data.append({
                "residue_number": residue_number,
                "RSA": rsa
            })

    df_rsa = pd.DataFrame(rsa_data)

    # --------------------------------------------------------
    # CONTACT COUNT
    # --------------------------------------------------------

    print()
    print("Calculating residue contacts...")

    parser3 = PDBParser(QUIET=True)

    structure3 = parser3.get_structure(
        "CLN3",
        PDB_FILE
    )

    atoms = list(structure3.get_atoms())

    ns = NeighborSearch(atoms)

    contact_data = []

    for chain in structure3[0]:
        for residue in chain:

            residue_number = residue.get_id()[1]

            if "CA" in residue:

                ca = residue["CA"]

                neighbors = ns.search(
                    ca.get_vector().get_array(),
                    CONTACT_RADIUS
                )

                neighbor_residues = set(
                    (
                        atom.get_parent().get_id()[1]
                    )
                    for atom in neighbors
                    if (
                        atom.get_parent().get_id()[1]
                        != residue_number
                    )
                )

                contact_data.append({
                    "residue_number": residue_number,
                    "contact_count": len(neighbor_residues)
                })

    df_contacts = pd.DataFrame(contact_data)

    # --------------------------------------------------------
    # MERGE STRUCTURAL FEATURES
    # --------------------------------------------------------

    df_final = df_plddt.merge(
        df_rsa,
        on="residue_number",
        how="left"
    )

    df_final = df_final.merge(
        df_contacts,
        on="residue_number",
        how="left"
    )

    # --------------------------------------------------------
    # BURIAL
    # --------------------------------------------------------

    df_final["burial"] = df_final["RSA"].apply(
        lambda x:
            "buried" if pd.notna(x) and x < 0.2
            else (
                "exposed" if pd.notna(x) and x > 0.5
                else (
                    "intermediate" if pd.notna(x)
                    else np.nan
                )
            )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    df_final.to_csv(
        PDB_OUTPUT_FILE,
        index=False
    )

    print()
    print("Structural features saved:")
    print(PDB_OUTPUT_FILE)
    print("Shape:", df_final.shape)

    return df_final


# ============================================================
# EXTRACT RESIDUE POSITION
# ============================================================

def extract_position_from_name(name):

    name = str(name)

    # p.Glu295Lys
    match = re.search(
        r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})',
        name
    )

    if match:

        position = int(match.group(2))

        alt_aa = THREE_TO_ONE.get(
            match.group(3),
            None
        )

        return position, alt_aa

    # p.Glu295Ter
    match = re.search(
        r'p\.([A-Z][a-z]{2})(\d+)Ter',
        name
    )

    if match:
        return int(match.group(2)), None

    # p.Glu295fs
    match = re.search(
        r'p\.([A-Z][a-z]{2})(\d+)fs',
        name
    )

    if match:
        return int(match.group(2)), None

    # p.T294M
    match = re.search(
        r'p\.([A-Z])(\d+)([A-Z])',
        name
    )

    if match:
        return int(match.group(2)), match.group(3)

    # p.T294*
    match = re.search(
        r'p\.([A-Z])(\d+)\*',
        name
    )

    if match:
        return int(match.group(2)), None

    # c.XXXA>G
    match = re.search(
        r'c\.(\d+)[ACGT]>[ACGT]',
        name
    )

    if match:

        nucleotide_position = int(match.group(1))

        amino_acid_position = (
            nucleotide_position + 2
        ) // 3

        return amino_acid_position, None

    # c.XXXdel / ins / dup
    match = re.search(
        r'c\.(\d+)(?:del|ins|dup)',
        name
    )

    if match:

        nucleotide_position = int(match.group(1))

        amino_acid_position = (
            nucleotide_position + 2
        ) // 3

        return amino_acid_position, None

    return None, None


# ============================================================
# LOAD ALPHAMISSENSE
# ============================================================

def load_alphamissense():

    print()
    print("Loading AlphaMissense...")

    if not os.path.exists(AM_FILE):
        raise FileNotFoundError(
            "\nAlphaMissense file not found:\n" + AM_FILE
        )

    am = pd.read_csv(
        AM_FILE,
        sep="\t"
    )

    required_columns = [
        "position",
        "a.a.2",
        "pathogenicity score",
        "pathogenicity class"
    ]

    missing = [
        col for col in required_columns
        if col not in am.columns
    ]

    if missing:
        raise ValueError(
            "\nMissing AlphaMissense columns:\n"
            + str(missing)
        )

    am = am[required_columns].copy()

    am.columns = [
        "residue_position",
        "alt_aa",
        "am_pathogenicity",
        "am_class"
    ]

    am["residue_position"] = pd.to_numeric(
        am["residue_position"],
        errors="coerce"
    )

    print("AlphaMissense entries:", len(am))

    return am


# ============================================================
# LOAD STRUCTURAL FEATURES
# ============================================================

def load_structural_features():

    if not os.path.exists(PDB_OUTPUT_FILE):
        run_structural_feature_extraction()

    pdb = pd.read_csv(PDB_OUTPUT_FILE)

    columns = [
        "residue_number",
        "pLDDT",
        "secondary_structure",
        "RSA",
        "burial"
    ]

    if "contact_count" in pdb.columns:
        columns.append("contact_count")

    pdb = pdb[columns].copy()

    pdb = pdb.rename(
        columns={
            "residue_number": "residue_position"
        }
    )

    pdb["residue_position"] = pd.to_numeric(
        pdb["residue_position"],
        errors="coerce"
    )

    return pdb


# ============================================================
# CADD-INDEL QUERY
# ============================================================

def query_cadd_indel(chrom, pos, ref, alt):

    try:

        region = f"{chrom}:{pos}-{pos}"

        cmd = [
            "wsl",
            "bash",
            "-c",
            f"tabix {CADD_INDEL_FILE} {region}"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        for line in result.stdout.strip().split("\n"):

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) >= 6:

                if parts[2] == ref and parts[3] == alt:

                    return (
                        float(parts[4]),
                        float(parts[5])
                    )

    except Exception:
        pass

    return None, None


# ============================================================
# CADD-INDEL FEATURES
# ============================================================

def get_cadd_indel_scores(df_indels):

    print()
    print("Querying CADD-Indel...")

    raw_scores = []
    phred_scores = []

    total = len(df_indels)

    for i, (_, row) in enumerate(df_indels.iterrows()):

        if i % 50 == 0:
            print(f"Progress: {i}/{total}")

        chrom = str(
            row["Chromosome"]
        ).replace("chr", "")

        position = int(row["Start"])

        ref = str(row["ReferenceAllele"])
        alt = str(row["AlternateAllele"])

        raw, phred = query_cadd_indel(
            chrom,
            position,
            ref,
            alt
        )

        raw_scores.append(raw)
        phred_scores.append(phred)

    df_indels = df_indels.copy()

    df_indels["CADD_indel_raw"] = raw_scores
    df_indels["CADD_indel_phred"] = phred_scores

    print()
    print(
        "CADD-Indel matches:",
        sum(x is not None for x in phred_scores),
        "/",
        total
    )

    return df_indels


# ============================================================
# LOAD AND FILTER CLINVAR
# ============================================================

def load_and_filter_clinvar():

    print()
    print("=" * 75)
    print("LOADING CLINVAR 2026")
    print("=" * 75)

    if not os.path.exists(CLINVAR_FILE):
        raise FileNotFoundError(
            "\nClinVar 2026 file not found:\n" + CLINVAR_FILE
        )

    df = pd.read_csv(
        CLINVAR_FILE,
        sep="\t",
        low_memory=False
    )

    print()
    print("Original ClinVar shape:", df.shape)

    # --------------------------------------------------------
    # COLUMN CLEANING
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
        .str.replace("#", "", regex=False)
    )

    required_columns = [
        "VariationID",
        "GeneSymbol",
        "ClinicalSignificance",
        "Chromosome",
        "PositionVCF",
        "Assembly",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "Name"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "\nRequired ClinVar columns missing:\n"
            + str(missing)
        )

    df = df[required_columns].copy()

    # --------------------------------------------------------
    # RENAME
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "ReferenceAlleleVCF": "ReferenceAllele",
            "AlternateAlleleVCF": "AlternateAllele",
            "PositionVCF": "Start"
        }
    )

    # --------------------------------------------------------
    # CLN3 FILTER
    # --------------------------------------------------------

    df = df[
        df["GeneSymbol"].astype(str).str.strip() == GENE
    ].copy()

    print()
    print("CLN3 variants:", len(df))

    # --------------------------------------------------------
    # GRCh38 FILTER
    # --------------------------------------------------------

    df = df[
        df["Assembly"].astype(str).str.strip() == "GRCh38"
    ].copy()

    print("GRCh38 variants:", len(df))

    # --------------------------------------------------------
    # CLINICAL SIGNIFICANCE
    # --------------------------------------------------------

    df["ClinicalSignificance"] = (
        df["ClinicalSignificance"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # --------------------------------------------------------
    # NUMERIC COORDINATES
    # --------------------------------------------------------

    df["Start"] = pd.to_numeric(
        df["Start"],
        errors="coerce"
    )

    df = df[df["Start"].notna() & (df["Start"] > 0)].copy()

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    df = df.drop_duplicates(subset="VariationID")

    print("After duplicate removal:", len(df))

    return df


# ============================================================
# CREATE LABELLED AND VUS SPLITS
# ============================================================

def split_labelled_and_vus(df):

    print()
    print("=" * 75)
    print("SPLITTING LABELLED VARIANTS AND VUS")
    print("=" * 75)

    # --------------------------------------------------------
    # LABELLED VARIANTS
    # --------------------------------------------------------

    labelled_values = [
        "pathogenic",
        "likely pathogenic",
        "benign",
        "likely benign"
    ]

    labelled = df[
        df["ClinicalSignificance"].isin(labelled_values)
    ].copy()

    labelled["label"] = (
        labelled["ClinicalSignificance"]
        .map({
            "pathogenic": 1,
            "likely pathogenic": 1,
            "benign": 0,
            "likely benign": 0
        })
    )

    # --------------------------------------------------------
    # VUS
    # --------------------------------------------------------

    vus = df[
        df["ClinicalSignificance"] == "uncertain significance"
    ].copy()

    print()
    print("Labelled variants:", len(labelled))
    print("VUS variants:", len(vus))

    print()
    print("Labelled distribution:")

    if len(labelled) > 0:
        print(
            labelled["ClinicalSignificance"]
            .value_counts()
            .to_string()
        )

    return labelled, vus


# ============================================================
# COMMON ANNOTATION PIPELINE
# ============================================================

def annotate_dataset(df, dataset_name, annotation_prefix):

    print()
    print("=" * 75)
    print("ANNOTATING:", dataset_name)
    print("=" * 75)

    df = df.copy()

    if df.empty:
        print("Dataset is empty. Skipping annotation.")
        return df

    # --------------------------------------------------------
    # EXTRACT RESIDUE POSITION
    # --------------------------------------------------------

    print()
    print("Extracting variant positions...")

    extracted = df["Name"].apply(
        extract_position_from_name
    )

    df["residue_position"] = extracted.apply(
        lambda x: x[0]
    )

    df["alt_aa"] = extracted.apply(
        lambda x: x[1]
    )

    # --------------------------------------------------------
    # CLEAN ALLELES
    # --------------------------------------------------------

    df["ReferenceAllele"] = (
        df["ReferenceAllele"]
        .astype(str)
        .str.strip()
    )

    df["AlternateAllele"] = (
        df["AlternateAllele"]
        .astype(str)
        .str.strip()
    )

    df["Chromosome"] = (
        df["Chromosome"]
        .astype(str)
        .str.strip()
        .str.replace(r"^chr", "", regex=True)
    )

    # --------------------------------------------------------
    # REMOVE INVALID VARIANTS
    # --------------------------------------------------------

    df = df[
        df["ReferenceAllele"].notna()
        & df["AlternateAllele"].notna()
        & (df["ReferenceAllele"].str.lower() != "na")
        & (df["AlternateAllele"].str.lower() != "na")
        & (df["ReferenceAllele"] != ".")
        & (df["AlternateAllele"] != ".")
        & df["Start"].notna()
        & (df["Start"] > 0)
    ].copy()

    if df.empty:
        print("No valid variants remain.")
        return df

    # --------------------------------------------------------
    # VARIANT TYPE
    # --------------------------------------------------------

    is_snp = (
        (df["ReferenceAllele"].str.len() == 1)
        & (df["AlternateAllele"].str.len() == 1)
    )

    df_snps = df[is_snp].copy()
    df_indels = df[~is_snp].copy()

    print()
    print("SNPs:", len(df_snps))
    print("Indels:", len(df_indels))

    # --------------------------------------------------------
    # ANNOVAR
    # --------------------------------------------------------

    if len(df_snps) > 0:

        print()
        print("Running ANNOVAR for", dataset_name)

        df_snps["Chromosome"] = (
            "chr" + df_snps["Chromosome"].astype(str)
        )

        annovar_input = os.path.join(
            RANKING_DIR,
            annotation_prefix + "_annovar.avinput"
        )

        annovar_output_prefix = os.path.join(
            RANKING_DIR,
            annotation_prefix
        )

        annovar_df = df_snps[
            [
                "Chromosome",
                "Start",
                "Start",
                "ReferenceAllele",
                "AlternateAllele"
            ]
        ].copy()

        annovar_df.columns = [
            "chr",
            "start",
            "end",
            "ref",
            "alt"
        ]

        annovar_df.to_csv(
            annovar_input,
            sep="\t",
            header=False,
            index=False
        )

        # Windows path -> WSL path
        wsl_input = (
            annovar_input
            .replace("D:", "/mnt/d")
            .replace("\\", "/")
        )

        wsl_output = (
            annovar_output_prefix
            .replace("D:", "/mnt/d")
            .replace("\\", "/")
        )

        # ----------------------------------------------------
        # ANNOVAR COMMAND
        # ----------------------------------------------------

        command = (
            f"cd {WSL_ANNOVAR_DIR} && "
            f"./table_annovar.pl "
            f"{wsl_input} "
            f"humandb/ "
            f"-buildver hg38 "
            f"-out {wsl_output} "
            f"-remove "
            f"-protocol refGene,avsnp150,dbnsfp42a "
            f"-operation g,f,f "
            f"-nastring . "
            f"-csvout "
            f"-otherinfo"
        )

        subprocess.run(
            [
                "wsl",
                "bash",
                "-c",
                command
            ],
            check=True
        )

        annovar_file = (
            annovar_output_prefix
            + ".hg38_multianno.csv"
        )

        if not os.path.exists(annovar_file):
            raise FileNotFoundError(
                "\nANNOVAR output not found:\n" + annovar_file
            )

        df_ann = pd.read_csv(
            annovar_file,
            low_memory=False
        )

        # ----------------------------------------------------
        # CLEAN ANNOVAR COORDINATES
        # ----------------------------------------------------

        df_snps["Start"] = pd.to_numeric(
            df_snps["Start"],
            errors="coerce"
        )

        df_ann["Start"] = pd.to_numeric(
            df_ann["Start"],
            errors="coerce"
        )

        # ----------------------------------------------------
        # MERGE ANNOVAR
        # ----------------------------------------------------

        merge_keys_right = [
            "Chr",
            "Start",
            "Ref",
            "Alt"
        ]

        missing_keys = [
            col for col in merge_keys_right
            if col not in df_ann.columns
        ]

        if missing_keys:
            raise ValueError(
                "ANNOVAR output is missing merge columns: "
                + str(missing_keys)
            )

        df_snps = df_snps.merge(
            df_ann,
            left_on=[
                "Chromosome",
                "Start",
                "ReferenceAllele",
                "AlternateAllele"
            ],
            right_on=[
                "Chr",
                "Start",
                "Ref",
                "Alt"
            ],
            how="left",
            suffixes=("", "_ann")
        )

        print()
        print("ANNOVAR annotation complete.")

    # --------------------------------------------------------
    # CADD-INDEL
    # --------------------------------------------------------

    if len(df_indels) > 0:

        print()
        print("Running CADD-Indel for", dataset_name)

        df_indels["Chromosome"] = (
            "chr" + df_indels["Chromosome"].astype(str)
        )

        df_indels = get_cadd_indel_scores(df_indels)

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    df = pd.concat(
        [
            df_snps,
            df_indels
        ],
        ignore_index=True,
        sort=False
    )

    # --------------------------------------------------------
    # VARIANT LENGTH
    # --------------------------------------------------------

    df["var_length"] = (
        df["AlternateAllele"].str.len()
        - df["ReferenceAllele"].str.len()
    )

    # --------------------------------------------------------
    # ALPHAMISSENSE
    # --------------------------------------------------------

    print()
    print("Merging AlphaMissense...")

    am = load_alphamissense()

    df = df.merge(
        am,
        on=[
            "residue_position",
            "alt_aa"
        ],
        how="left"
    )

    df["am_likely_pathogenic"] = (
        df["am_class"] == "likely_pathogenic"
    ).astype(int)

    df["am_ambiguous"] = (
        df["am_class"] == "ambiguous"
    ).astype(int)

    # --------------------------------------------------------
    # ALPHAFOLD
    # --------------------------------------------------------

    print()
    print("Merging AlphaFold features...")

    pdb = load_structural_features()

    df = df.merge(
        pdb,
        on="residue_position",
        how="left"
    )

    # --------------------------------------------------------
    # STRUCTURAL ENCODING
    # --------------------------------------------------------

    df["ss_helix"] = (
        df["secondary_structure"] == "H"
    ).astype(int)

    df["ss_sheet"] = (
        df["secondary_structure"] == "E"
    ).astype(int)

    df["ss_coil"] = (
        df["secondary_structure"] == "-"
    ).astype(int)

    df["is_buried"] = (
        df["burial"] == "buried"
    ).astype(int)

    df["is_exposed"] = (
        df["burial"] == "exposed"
    ).astype(int)

    df["is_intermediate"] = (
        df["burial"] == "intermediate"
    ).astype(int)

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = [
        "LINSIGHT",
        "GERP++_RS",
        "phyloP100way_vertebrate",
        "phastCons100way_vertebrate",
        "SiPhy_29way_logOdds",
        "SIFT_score",
        "Polyphen2_HDIV_score",
        "Polyphen2_HVAR_score",
        "MutationTaster_score",
        "MetaRNN_score",
        "VEST4_score",
        "REVEL_score",
        "CADD_indel_raw",
        "CADD_indel_phred",
        "var_length",
        "am_pathogenicity",
        "am_likely_pathogenic",
        "am_ambiguous",
        "pLDDT",
        "RSA",
        "ss_helix",
        "ss_sheet",
        "ss_coil",
        "is_buried",
        "is_exposed",
        "is_intermediate",
        "contact_count"
    ]

    features = [
        feature for feature in features
        if feature in df.columns
    ]

    print()
    print("Features available:", len(features))

    if not features:
        raise ValueError(
            "No requested ML features were found. "
            "Check ANNOVAR/dbNSFP column names and annotations."
        )

    # --------------------------------------------------------
    # CLEAN FEATURES
    # --------------------------------------------------------

    for feature in features:

        df[feature] = df[feature].replace(
            ".",
            np.nan
        )

        df[feature] = df[feature].map(
            lambda value:
                value.split(";")[0]
                if isinstance(value, str) and ";" in value
                else value
        )

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    # --------------------------------------------------------
    # MISSING-VALUE FILTER
    #
    # These are fixed rules based on feature count.
    # They are not thresholds learned from either dataset.
    # The same rules are applied to labelled and VUS data.
    # --------------------------------------------------------

    is_snp_row = df["var_length"] == 0
    is_indel_row = df["var_length"] != 0

    df_snps_final = df[is_snp_row].copy()
    df_indels_final = df[is_indel_row].copy()

    snp_threshold = max(
        1,
        int(len(features) * 0.5)
    )

    indel_threshold = max(
        1,
        int(len(features) * 0.3)
    )

    df_snps_final = df_snps_final.dropna(
        subset=features,
        thresh=snp_threshold
    )

    df_indels_final = df_indels_final.dropna(
        subset=features,
        thresh=indel_threshold
    )

    # --------------------------------------------------------
    # FINAL DATASET
    # --------------------------------------------------------

    df = pd.concat(
        [
            df_snps_final,
            df_indels_final
        ],
        ignore_index=True,
        sort=False
    )

    # --------------------------------------------------------
    # IDENTIFIER COLUMNS
    # --------------------------------------------------------

    identifiers = [
        "VariationID",
        "GeneSymbol",
        "ClinicalSignificance",
        "Chromosome",
        "Start",
        "ReferenceAllele",
        "AlternateAllele",
        "Name",
        "Assembly",
        "residue_position",
        "alt_aa"
    ]

    identifiers = [
        column for column in identifiers
        if column in df.columns
    ]

    # --------------------------------------------------------
    # FINAL COLUMNS
    # --------------------------------------------------------

    if "label" in df.columns:

        final_columns = (
            identifiers
            + features
            + ["label"]
        )

    else:

        final_columns = (
            identifiers
            + features
        )

    # Remove duplicated column names while preserving order.
    final_columns = list(dict.fromkeys(final_columns))

    final_columns = [
        column for column in final_columns
        if column in df.columns
    ]

    df = df[final_columns].copy()

    print()
    print("Final", dataset_name, "shape:", df.shape)

    if "label" in df.columns:
        print("Final label distribution:")
        print(df["label"].value_counts(dropna=False).to_string())

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("CLN3 VUS PREPROCESSING + ANNOTATION")
    print("=" * 80)

    print()
    print("CLINVAR SOURCE:")
    print(CLINVAR_FILE)

    print()
    print("OUTPUT DIRECTORY:")
    print(RANKING_DIR)

    # --------------------------------------------------------
    # LOAD 2026 CLINVAR SOURCE
    # --------------------------------------------------------

    all_cln3 = load_and_filter_clinvar()

    # --------------------------------------------------------
    # SPLIT LABELLED AND VUS
    # --------------------------------------------------------

    labelled, vus = split_labelled_and_vus(
        all_cln3
    )

    # --------------------------------------------------------
    # ANNOTATE LABELLED 2026 DATA
    # --------------------------------------------------------

    labelled_processed = annotate_dataset(
        labelled,
        "LABELLED TRAINING DATA 2026",
        "CLN3_2026"
    )

    # --------------------------------------------------------
    # ANNOTATE VUS FROM THE SAME 2026 SOURCE
    # --------------------------------------------------------

    vus_processed = annotate_dataset(
        vus,
        "VUS DATA 2026",
        "CLN3_VUS"
    )

    # --------------------------------------------------------
    # SAVE LABELLED DATA
    # --------------------------------------------------------

    labelled_processed.to_csv(
        LABELED_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # SAVE VUS DATA
    # --------------------------------------------------------

    vus_processed.to_csv(
        VUS_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CLN3 PREPROCESSING + ANNOTATION COMPLETE")
    print("=" * 80)

    print()
    print("Labelled training dataset:")
    print(LABELED_OUTPUT)

    print()
    print("VUS dataset:")
    print(VUS_OUTPUT)

    print()
    print("Training shape:", labelled_processed.shape)
    print("VUS shape:", vus_processed.shape)

    print()
    print("Labelled class distribution:")

    if (
        not labelled_processed.empty
        and "label" in labelled_processed.columns
    ):
        print(
            labelled_processed["label"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    print()
    print("All generated files are stored in:")
    print(RANKING_DIR)

    print()
    print("=" * 80)
