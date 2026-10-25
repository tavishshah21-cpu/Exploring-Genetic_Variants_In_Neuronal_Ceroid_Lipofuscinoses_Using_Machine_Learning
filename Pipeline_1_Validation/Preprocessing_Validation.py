
# ============================================================
# CLN3 GNOMAD EXTERNAL CANDIDATE PREPROCESSING PIPELINE
# ============================================================
# Adapted from Preprocessingv3(2).py
#
# Input:
#   Pipeline_1_Validation/
#       CLN3_gnomAD_external_candidates_original_columns.csv
#
# Steps:
#   1. Load external gnomAD candidates
#   2. Validate genomic coordinates and alleles
#   3. Extract protein positions
#   4. Separate SNPs and indels
#   5. ANNOVAR + dbNSFP annotation for SNPs
#   6. CADD-Indel lookup for indels
#   7. AlphaMissense integration
#   8. AlphaFold structural feature extraction
#   9. Feature engineering
#  10. Save processed candidates and original metadata
#
# External candidates remain UNLABELLED.
# Missing prediction scores remain missing.
# ============================================================

import os
import re
import subprocess

import numpy as np
import pandas as pd
import pydssp

from Bio.PDB import PDBParser, ShrakeRupley, NeighborSearch


# ============================================================
# CONFIGURATION
# ============================================================

GENE = "CLN3"

BASE_DIR = (
    r"D:\tavish_shah\PES\capstone"
    r"\Pipeline_1_Validation"
)

STATISTICS_DIR = (
    r"D:\tavish_shah\PES\capstone"
    r"\Pipeline_1_Statistics"
)

GNOMAD_FILE = os.path.join(
    BASE_DIR,
    "CLN3_gnomAD_external_candidates_original_columns.csv"
)

WSL_ANNOVAR_DIR = "~/annovar_work/annovar"

CADD_INDEL_FILE = (
    "~/cadd_data/gnomad.genomes.r4.0.indel.tsv.gz"
)

AM_FILE = os.path.join(
    STATISTICS_DIR,
    "AlphaMissense-Search-Q13286.tsv"
)

PDB_FILE = os.path.join(
    STATISTICS_DIR,
    "AF-Q13286-F1-model_v6.pdb"
)

PDB_OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "cln3_structural_features.csv"
)

OUTPUT_PREFIX = os.path.join(
    BASE_DIR,
    "CLN3_gnomAD_external_candidates"
)

CONTACT_RADIUS = 8.0


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
# STRUCTURAL FEATURE EXTRACTION
# ============================================================

def run_structural_feature_extraction():

    print("\n" + "=" * 70)
    print("CLN3 STRUCTURAL FEATURE EXTRACTION")
    print("=" * 70)

    if not os.path.exists(PDB_FILE):
        raise FileNotFoundError(
            f"AlphaFold PDB not found: {PDB_FILE}"
        )

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("CLN3", PDB_FILE)
    model = structure[0]

    # --------------------------------------------------------
    # Extract pLDDT
    # --------------------------------------------------------

    plddt_data = []

    for chain in model:
        for residue in chain:

            if "CA" not in residue:
                continue

            plddt_data.append({
                "residue_number": residue.get_id()[1],
                "amino_acid": residue.get_resname(),
                "pLDDT": residue["CA"].get_bfactor()
            })

    df_plddt = pd.DataFrame(plddt_data)

    if df_plddt.empty:
        raise ValueError(
            "No CA atoms found in the AlphaFold PDB."
        )

    print("Residues extracted:", len(df_plddt))

    # --------------------------------------------------------
    # Secondary structure
    # --------------------------------------------------------

    print("Calculating secondary structure...")

    with open(PDB_FILE, "r") as handle:
        pdb_text = handle.read()

    coords = pydssp.read_pdbtext(pdb_text)

    ss_results = pydssp.assign(
        coords,
        out_type="c3"
    )

    ss_values = list(ss_results)

    if len(ss_values) != len(df_plddt):
        raise ValueError(
            "Secondary-structure results do not match "
            "the extracted residue count. Check PDB "
            "chain and residue handling."
        )

    df_plddt["secondary_structure"] = ss_values

    print(
        "Helix:",
        (df_plddt["secondary_structure"] == "H").sum()
    )

    print(
        "Sheet:",
        (df_plddt["secondary_structure"] == "E").sum()
    )

    print(
        "Coil:",
        (df_plddt["secondary_structure"] == "-").sum()
    )

    # --------------------------------------------------------
    # Relative solvent accessibility (RSA)
    # --------------------------------------------------------

    print("Calculating RSA...")

    structure2 = parser.get_structure(
        "CLN3_RSA",
        PDB_FILE
    )

    sr = ShrakeRupley()
    sr.compute(structure2, level="R")

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

            asa = getattr(residue, "sasa", np.nan)
            max_value = max_asa.get(residue_name, 200)

            rsa = (
                round(asa / max_value, 3)
                if pd.notna(asa) and max_value > 0
                else np.nan
            )

            rsa_data.append({
                "residue_number": residue_number,
                "RSA": rsa
            })

    df_rsa = pd.DataFrame(rsa_data)

    # --------------------------------------------------------
    # Contact count
    # --------------------------------------------------------

    print("Calculating residue contacts...")

    structure3 = parser.get_structure(
        "CLN3_CONTACTS",
        PDB_FILE
    )

    atoms = list(structure3.get_atoms())
    ns = NeighborSearch(atoms)

    contact_data = []

    for chain in structure3[0]:
        for residue in chain:

            if "CA" not in residue:
                continue

            residue_number = residue.get_id()[1]
            ca = residue["CA"]

            neighbors = ns.search(
                ca.get_coord(),
                CONTACT_RADIUS
            )

            neighbor_residues = {
                (
                    atom.get_parent().get_parent().id,
                    atom.get_parent().get_id()[1]
                )
                for atom in neighbors
                if (
                    atom.get_parent().get_id()[1]
                    != residue_number
                )
            }

            contact_data.append({
                "residue_number": residue_number,
                "contact_count": len(neighbor_residues)
            })

    df_contacts = pd.DataFrame(contact_data)

    # --------------------------------------------------------
    # Merge structural features
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

    df_final["burial"] = df_final["RSA"].apply(
        lambda x:
            "buried"
            if pd.notna(x) and x < 0.2
            else (
                "exposed"
                if pd.notna(x) and x > 0.5
                else (
                    "intermediate"
                    if pd.notna(x)
                    else np.nan
                )
            )
    )

    df_final.to_csv(
        PDB_OUTPUT_FILE,
        index=False
    )

    print("Structural features saved:", PDB_OUTPUT_FILE)
    print("Structural feature shape:", df_final.shape)

    return df_final


# ============================================================
# EXTRACT PROTEIN POSITION AND ALTERNATE AMINO ACID
# ============================================================

def extract_position_from_name(name):

    name = str(name)

    # Three-letter missense: p.Glu295Lys
    match = re.search(
        r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})",
        name
    )

    if match:
        return (
            int(match.group(2)),
            THREE_TO_ONE.get(match.group(3))
        )

    # Three-letter stop: p.Glu295Ter
    match = re.search(
        r"p\.([A-Z][a-z]{2})(\d+)Ter",
        name
    )

    if match:
        return int(match.group(2)), None

    # Frameshift
    match = re.search(
        r"p\.([A-Z][a-z]{2})(\d+)fs",
        name
    )

    if match:
        return int(match.group(2)), None

    # Single-letter missense: p.T294M
    match = re.search(
        r"p\.([A-Z])(\d+)([A-Z])",
        name
    )

    if match:
        return int(match.group(2)), match.group(3)

    # Single-letter stop: p.T294*
    match = re.search(
        r"p\.([A-Z])(\d+)\*",
        name
    )

    if match:
        return int(match.group(2)), None

    # Coding substitution: c.123A>G
    match = re.search(
        r"c\.(\d+)[ACGT]>[ACGT]",
        name
    )

    if match:
        nt_position = int(match.group(1))
        return (nt_position + 2) // 3, None

    # Coding deletion, insertion or duplication
    match = re.search(
        r"c\.(\d+)(?:del|ins|dup)",
        name
    )

    if match:
        nt_position = int(match.group(1))
        return (nt_position + 2) // 3, None

    return None, None


# ============================================================
# LOAD ALPHAMISSENSE
# ============================================================

def load_alphamissense():

    print("\nLoading AlphaMissense...")

    if not os.path.exists(AM_FILE):
        raise FileNotFoundError(
            f"AlphaMissense file not found: {AM_FILE}"
        )

    am = pd.read_csv(
        AM_FILE,
        sep="\t",
        low_memory=False
    )

    required = [
        "position",
        "a.a.2",
        "pathogenicity score",
        "pathogenicity class"
    ]

    missing = [
        col for col in required
        if col not in am.columns
    ]

    if missing:
        raise ValueError(
            f"AlphaMissense file is missing columns: {missing}"
        )

    am = am[required].copy()

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

    am["alt_aa"] = (
        am["alt_aa"]
        .astype("string")
        .str.strip()
    )

    am["am_pathogenicity"] = pd.to_numeric(
        am["am_pathogenicity"],
        errors="coerce"
    )

    print("AlphaMissense entries:", len(am))

    return am


# ============================================================
# LOAD STRUCTURAL FEATURES
# ============================================================

def load_structural_features():

    run_structural_feature_extraction()

    pdb = pd.read_csv(
        PDB_OUTPUT_FILE,
        low_memory=False
    )

    columns = [
        "residue_number",
        "pLDDT",
        "secondary_structure",
        "RSA",
        "burial",
        "contact_count"
    ]

    pdb = pdb[
        [col for col in columns if col in pdb.columns]
    ].copy()

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

        chrom = re.sub(
            r"^chr",
            "",
            str(chrom),
            flags=re.IGNORECASE
        )

        region = f"{chrom}:{pos}-{pos}"

        command = (
            f"tabix {CADD_INDEL_FILE} {region}"
        )

        result = subprocess.run(
            ["wsl", "bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=20
        )

        if result.returncode != 0:
            return None, None

        for line in result.stdout.strip().splitlines():

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 6:
                continue

            if (
                parts[2] == str(ref)
                and parts[3] == str(alt)
            ):
                try:
                    return float(parts[4]), float(parts[5])
                except (ValueError, TypeError):
                    return None, None

    except Exception as exc:

        print(
            f"CADD-Indel query failed for "
            f"{chrom}:{pos} {ref}>{alt}: {exc}"
        )

    return None, None


def get_cadd_indel_scores(df_indels):

    print(
        f"\nQuerying CADD-Indel for "
        f"{len(df_indels)} indel candidates..."
    )

    df_indels = df_indels.copy()

    raw_scores = []
    phred_scores = []

    for i, (_, row) in enumerate(df_indels.iterrows()):

        if i % 50 == 0:
            print(
                f"Progress: {i}/{len(df_indels)}"
            )

        try:
            pos = int(row["Start"])
        except (ValueError, TypeError):
            raw_scores.append(np.nan)
            phred_scores.append(np.nan)
            continue

        raw, phred = query_cadd_indel(
            row["Chromosome"],
            pos,
            row["ReferenceAllele"],
            row["AlternateAllele"]
        )

        raw_scores.append(raw)
        phred_scores.append(phred)

    df_indels["CADD_indel_raw"] = raw_scores
    df_indels["CADD_indel_phred"] = phred_scores

    print(
        "CADD-Indel matched:",
        pd.Series(phred_scores).notna().sum(),
        "/",
        len(df_indels)
    )

    return df_indels


# ============================================================
# LOAD EXTERNAL GNOMAD CANDIDATES
# ============================================================

def load_gnomad_candidates(input_file):

    print("\nLoading external gnomAD candidate CSV...")

    raw = pd.read_csv(
        input_file,
        low_memory=False
    )

    raw.columns = raw.columns.str.strip()

    required = [
        "gnomAD ID",
        "Chromosome",
        "Position",
        "Reference",
        "Alternate"
    ]

    missing = [
        col for col in required
        if col not in raw.columns
    ]

    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}"
        )

    raw = raw.copy()

    raw["_candidate_id"] = (
        raw["gnomAD ID"].astype(str)
    )

    df = pd.DataFrame(index=raw.index)

    df["_candidate_id"] = raw["_candidate_id"]

    df["VariationID"] = raw["gnomAD ID"]

    df["GeneSymbol"] = GENE

    df["Chromosome"] = (
        raw["Chromosome"]
        .astype(str)
        .str.strip()
        .str.replace(
            r"^chr",
            "",
            regex=True,
            case=False
        )
    )

    df["Start"] = pd.to_numeric(
        raw["Position"],
        errors="coerce"
    )

    df["Assembly"] = "GRCh38"

    df["ReferenceAllele"] = (
        raw["Reference"]
        .astype("string")
        .str.strip()
    )

    df["AlternateAllele"] = (
        raw["Alternate"]
        .astype("string")
        .str.strip()
    )

    # Prefer protein consequence, then transcript/HGVS.
    consequence_columns = [
        "Protein Consequence",
        "Transcript Consequence",
        "HGVS Consequence"
    ]

    available = [
        col for col in consequence_columns
        if col in raw.columns
    ]

    if available:

        name = raw[available[0]].copy()

        for col in available[1:]:
            name = name.fillna(raw[col])

        df["Name"] = name.fillna("").astype(str)

    else:
        df["Name"] = ""

    # Map original gnomAD columns to standard feature names.
    mapping = {
        "Allele Frequency": "Allele_Frequency",
        "Allele Count": "Allele_Count",
        "Allele Number": "Allele_Number",
        "Homozygote Count": "Homozygote_Count",
        "Hemizygote Count": "Hemizygote_Count",
        "cadd": "CADD_phred",
        "revel_max": "REVEL_score",
        "sift_max": "SIFT_score",
        "polyphen_max": "Polyphen2_HDIV_score",
        "phylop": "phyloP100way_vertebrate",
        "spliceai_ds_max": "spliceai_ds_max",
        "pangolin_largest_ds": "pangolin_largest_ds"
    }

    for source, target in mapping.items():

        if source in raw.columns:

            df[target] = pd.to_numeric(
                raw[source],
                errors="coerce"
            )

    # No artificial pathogenic/benign labels.
    df["label"] = np.nan

    print("Input candidates:", len(df))

    return df, raw


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def run_pipeline(input_file, output_prefix):

    print("\n" + "=" * 70)
    print("CLN3 GNOMAD EXTERNAL CANDIDATE PREPROCESSING")
    print("=" * 70)

    print("Input:", input_file)
    print("Output prefix:", output_prefix)

    os.makedirs(
        os.path.dirname(output_prefix),
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load candidates and original metadata
    # --------------------------------------------------------

    df, raw_metadata = load_gnomad_candidates(input_file)

    original_count = len(df)

    # --------------------------------------------------------
    # Validate coordinates and alleles
    # --------------------------------------------------------

    df["ReferenceAllele"] = (
        df["ReferenceAllele"]
        .astype("string")
        .str.strip()
    )

    df["AlternateAllele"] = (
        df["AlternateAllele"]
        .astype("string")
        .str.strip()
    )

    valid = (
        df["Chromosome"].notna()
        & df["Start"].notna()
        & (df["Start"] > 0)
        & df["ReferenceAllele"].notna()
        & df["AlternateAllele"].notna()
        & (df["ReferenceAllele"] != "")
        & (df["AlternateAllele"] != "")
        & (df["ReferenceAllele"] != ".")
        & (df["AlternateAllele"] != ".")
        & (df["ReferenceAllele"].str.lower() != "na")
        & (df["AlternateAllele"].str.lower() != "na")
    )

    df = df.loc[valid].copy()

    print(
        "After coordinate/allele validation:",
        len(df)
    )

    # --------------------------------------------------------
    # Extract protein positions
    # --------------------------------------------------------

    extracted = df["Name"].apply(
        extract_position_from_name
    )

    df["residue_position"] = extracted.apply(
        lambda x: x[0]
    )

    df["alt_aa"] = extracted.apply(
        lambda x: x[1]
    )

    df["residue_position"] = pd.to_numeric(
        df["residue_position"],
        errors="coerce"
    )

    print(
        "Candidates with protein positions:",
        df["residue_position"].notna().sum(),
        "/",
        len(df)
    )

    # --------------------------------------------------------
    # Separate SNPs and indels
    # --------------------------------------------------------

    df["var_length"] = (
        df["AlternateAllele"].str.len()
        - df["ReferenceAllele"].str.len()
    )

    is_snp = (
        (df["ReferenceAllele"].str.len() == 1)
        & (df["AlternateAllele"].str.len() == 1)
    )

    df_snps = df.loc[is_snp].copy()
    df_indels = df.loc[~is_snp].copy()

    print("SNPs:", len(df_snps))
    print("Indels:", len(df_indels))

    # --------------------------------------------------------
    # ANNOVAR + dbNSFP for SNPs
    # --------------------------------------------------------

    if not df_snps.empty:

        print("\nRunning ANNOVAR for SNPs...")

        df_snps["Chromosome"] = (
            "chr" + df_snps["Chromosome"].astype(str)
        )

        annovar_input = output_prefix + ".avinput"

        annovar_output_prefix = (
            output_prefix + "_annovar"
        )

        annovar_df = df_snps[
            [
                "Chromosome",
                "Start",
                "ReferenceAllele",
                "AlternateAllele"
            ]
        ].copy()

        annovar_df["Start"] = (
            annovar_df["Start"].astype(int)
        )

        annovar_df.insert(
            2,
            "End",
            annovar_df["Start"]
        )

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

        # Windows path -> WSL path.
        def to_wsl_path(path):

            path = os.path.abspath(path)

            if len(path) >= 2 and path[1] == ":":
                drive = path[0].lower()
                path = f"/mnt/{drive}" + path[2:]

            return path.replace("\\", "/")

        wsl_input = to_wsl_path(annovar_input)
        wsl_output = to_wsl_path(annovar_output_prefix)

        annovar_command = (
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
            f"-otherinfo "
            f"-arg '-splicing 5',,"
        )

        print("Executing ANNOVAR...")

        subprocess.run(
            ["wsl", "bash", "-lc", annovar_command],
            check=True
        )

        ann_file = (
            annovar_output_prefix
            + ".hg38_multianno.csv"
        )

        if not os.path.exists(ann_file):
            raise FileNotFoundError(
                f"ANNOVAR output was not created: {ann_file}"
            )

        df_ann = pd.read_csv(
            ann_file,
            low_memory=False
        )

        print("ANNOVAR rows:", len(df_ann))

        # Standardize chromosome names for matching.
        if "Chr" not in df_ann.columns:
            raise ValueError(
                "ANNOVAR output does not contain the Chr column."
            )

        for col in ["Start", "Ref", "Alt"]:
            if col not in df_ann.columns:
                raise ValueError(
                    f"ANNOVAR output is missing column: {col}"
                )

        df_snps["_chrom_key"] = (
            df_snps["Chromosome"]
            .astype(str)
            .str.replace(
                r"^chr",
                "",
                regex=True,
                case=False
            )
        )

        df_ann["_chrom_key"] = (
            df_ann["Chr"]
            .astype(str)
            .str.replace(
                r"^chr",
                "",
                regex=True,
                case=False
            )
        )

        df_snps["_merge_start"] = pd.to_numeric(
            df_snps["Start"],
            errors="coerce"
        )

        df_ann["_merge_start"] = pd.to_numeric(
            df_ann["Start"],
            errors="coerce"
        )

        # Use stable genomic keys and avoid duplicate
        # annotation-key columns in the merged result.
        df_snps["_merge_ref"] = (
            df_snps["ReferenceAllele"].astype(str)
        )

        df_snps["_merge_alt"] = (
            df_snps["AlternateAllele"].astype(str)
        )

        df_ann["_merge_ref"] = df_ann["Ref"].astype(str)
        df_ann["_merge_alt"] = df_ann["Alt"].astype(str)

        merge_keys = [
            "_chrom_key",
            "_merge_start",
            "_merge_ref",
            "_merge_alt"
        ]

        # Preserve the ANNOVAR columns as features, but
        # exclude redundant genomic key columns.
        ann_key_columns = [
            "Chr",
            "Start",
            "End",
            "Ref",
            "Alt",
            "_chrom_key",
            "_merge_start",
            "_merge_ref",
            "_merge_alt"
        ]

        annotation_columns = [
            col for col in df_ann.columns
            if col not in ann_key_columns
        ]

        df_ann_merge = df_ann[
            merge_keys + annotation_columns
        ].copy()

        df_snps = df_snps.merge(
            df_ann_merge,
            on=merge_keys,
            how="left",
            suffixes=("", "_ann")
        )

        df_snps.drop(
            columns=merge_keys,
            inplace=True,
            errors="ignore"
        )

        print(
            "SNPs after ANNOVAR merge:",
            len(df_snps)
        )

    else:
        print("No SNPs to annotate with ANNOVAR.")

    # --------------------------------------------------------
    # CADD-Indel for indels
    # --------------------------------------------------------

    if not df_indels.empty:

        print("\nProcessing indels using CADD-Indel...")

        df_indels["Chromosome"] = (
            "chr"
            + df_indels["Chromosome"]
            .astype(str)
            .str.replace(
                r"^chr",
                "",
                regex=True,
                case=False
            )
        )

        df_indels = get_cadd_indel_scores(df_indels)

    else:
        print("No indels to query.")

    # --------------------------------------------------------
    # Combine SNPs and indels
    # --------------------------------------------------------

    df = pd.concat(
        [df_snps, df_indels],
        ignore_index=True,
        sort=False
    )

    # --------------------------------------------------------
    # AlphaMissense integration
    # --------------------------------------------------------

    print("\nMerging AlphaMissense...")

    am = load_alphamissense()

    df["alt_aa"] = (
        df["alt_aa"].astype("string")
    )

    df = df.merge(
        am,
        on=["residue_position", "alt_aa"],
        how="left"
    )

    print(
        "AlphaMissense matches:",
        df["am_pathogenicity"].notna().sum(),
        "/",
        len(df)
    )

    # --------------------------------------------------------
    # CORRECTED ORIGINAL INDICATOR LOGIC
    # --------------------------------------------------------
    # These expressions intentionally match the original
    # Preprocessingv3 behavior.
    #
    # Direct comparisons return False for missing values,
    # so converting the Boolean results to integers is safe.
    # Missing AlphaMissense scores themselves remain missing.
    # --------------------------------------------------------

    df["am_likely_pathogenic"] = (
        df["am_class"] == "likely_pathogenic"
    ).astype(int)

    df["am_ambiguous"] = (
        df["am_class"] == "ambiguous"
    ).astype(int)

    # --------------------------------------------------------
    # AlphaFold structural features
    # --------------------------------------------------------

    print("\nMerging structural features...")

    pdb = load_structural_features()

    df = df.merge(
        pdb,
        on="residue_position",
        how="left"
    )

    print(
        "Structural matches:",
        df["pLDDT"].notna().sum(),
        "/",
        len(df)
    )

    # --------------------------------------------------------
    # Structural feature encoding
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
    # Feature selection
    # --------------------------------------------------------

    features = [
        # Conservation
        "LINSIGHT",
        "GERP++_RS",
        "phyloP100way_vertebrate",
        "phastCons100way_vertebrate",
        "SiPhy_29way_logOdds",

        # Functional prediction
        "SIFT_score",
        "Polyphen2_HDIV_score",
        "Polyphen2_HVAR_score",
        "MutationTaster_score",
        "MetaRNN_score",
        "VEST4_score",
        "REVEL_score",

        # CADD
        "CADD_phred",
        "CADD_indel_raw",
        "CADD_indel_phred",

        # gnomAD population features
        "Allele_Frequency",
        "Allele_Count",
        "Allele_Number",
        "Homozygote_Count",
        "Hemizygote_Count",

        # Additional prediction features
        "spliceai_ds_max",
        "pangolin_largest_ds",

        # Variant size
        "var_length",

        # AlphaMissense
        "am_pathogenicity",
        "am_likely_pathogenic",
        "am_ambiguous",

        # AlphaFold structural features
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

    # Add features only when they exist in the data.
    features = [
        col for col in features
        if col in df.columns
    ]

    print("\nFeatures available:", len(features))
    print(features)

    # --------------------------------------------------------
    # Convert selected features to numeric
    # --------------------------------------------------------

    df = df.replace(
        [".", "NA", "NaN", "nan", ""],
        np.nan
    )

    for col in features:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Preserve candidate metadata and all selected features
    # --------------------------------------------------------

    metadata_columns = [
        col for col in [
            "_candidate_id",
            "VariationID",
            "GeneSymbol",
            "Chromosome",
            "Start",
            "ReferenceAllele",
            "AlternateAllele",
            "Name",
            "residue_position",
            "alt_aa"
        ]
        if col in df.columns
    ]

    output_columns = list(dict.fromkeys(
        metadata_columns + features + ["label"]
    ))

    processed = df[output_columns].copy()

    # --------------------------------------------------------
    # Restore original gnomAD columns
    # --------------------------------------------------------

    original_metadata = raw_metadata.copy()

    original_metadata["_candidate_id"] = (
        original_metadata["_candidate_id"].astype(str)
    )

    processed["_candidate_id"] = (
        processed["_candidate_id"].astype(str)
    )

    # Include all original columns not already present.
    metadata_columns_to_add = [
        col for col in original_metadata.columns
        if (
            col != "_candidate_id"
            and col not in processed.columns
        )
    ]

    metadata_to_add = original_metadata[
        ["_candidate_id"] + metadata_columns_to_add
    ].copy()

    processed = processed.merge(
        metadata_to_add,
        on="_candidate_id",
        how="left",
        validate="many_to_one"
    )

    processed.drop(
        columns=["_candidate_id"],
        inplace=True,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Save processed candidates
    # --------------------------------------------------------

    output_file = (
        output_prefix
        + "_CLN3_ML_READY.csv"
    )

    processed.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GNOMAD PREPROCESSING COMPLETED")
    print("=" * 70)

    print("Original candidates:", original_count)
    print("Final candidate rows:", len(processed))
    print("Final dataset shape:", processed.shape)
    print("Feature count:", len(features))
    print("Output:", output_file)

    print(
        "Candidates with known labels:",
        processed["label"].notna().sum()
    )

    print(
        "Candidates without labels:",
        processed["label"].isna().sum()
    )

    print("\nFirst five output rows:")
    print(processed.head().to_string(index=False))

    return processed


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    if not os.path.exists(GNOMAD_FILE):
        raise FileNotFoundError(
            "External gnomAD CSV not found:\n"
            f"{GNOMAD_FILE}"
        )

    run_pipeline(
        GNOMAD_FILE,
        OUTPUT_PREFIX
    )
