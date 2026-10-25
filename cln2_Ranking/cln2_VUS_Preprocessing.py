# ============================================================
# CLN2 INTEGRATED PREPROCESSING PIPELINE
# ============================================================
#
# Gene: TPP1 (CLN2)
#
# Input:
#   Pipeline_1_Statistics/variant_summary_2026.txt
#
# Processing:
#   1. Filter ClinVar records for TPP1 and GRCh38
#   2. Create labelled pathogenic/benign dataset
#   3. Create a separate VUS dataset
#   4. Annotate SNPs using ANNOVAR and dbNSFP
#   5. Query CADD-Indel for indels
#   6. Merge AlphaMissense features
#   7. Extract structural features
#   8. Engineer features and filter missing values
#   9. Save ML-ready CSV files
#
# Important:
#   VUS records are not used as labelled training examples.
#   This script performs preprocessing only.
#
# ============================================================

import os
import re
import shlex
import subprocess
from pathlib import Path

import pandas as pd
import pydssp

from Bio.PDB import (
    PDBParser,
    ShrakeRupley,
    NeighborSearch,
)


# ============================================================
# CONFIGURATION
# ============================================================

GENE = "TPP1"

PROJECT_DIR = Path(
    r"D:\tavish_shah\PES\capstone"
)

CLINVAR_DIR = PROJECT_DIR / "Pipeline_1_Statistics"
DATA_DIR = PROJECT_DIR / "Pipeline_1_cln2"
OUTPUT_DIR = PROJECT_DIR / "cln2_Ranking"

CLINVAR_2026 = CLINVAR_DIR / "variant_summary_2026.txt"

ALPHAMISSENSE_FILE = (
    DATA_DIR / "AlphaMissense-Search-O14773.tsv"
)

PDB_FILE = DATA_DIR / "AF-O14773-F1-model_v6.pdb"

STRUCTURAL_OUTPUT = (
    DATA_DIR / "cln2_structural_features.csv"
)

# WSL paths. The directory must actually exist inside WSL.
WSL_ANNOVAR_DIR = "$HOME/annovar_work/annovar"

WSL_CADD_FILE = (
    "$HOME/cadd_data/gnomad.genomes.r4.0.indel.tsv.gz"
)

CONTACT_RADIUS = 8.0

# SNPs must have at least this fraction of feature values.
SNP_MIN_FEATURE_COVERAGE = 0.50

# Indels must have at least this fraction of feature values.
INDEL_MIN_FEATURE_COVERAGE = 0.30

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


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
    "Val": "V",
}


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [
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
    "contact_count",
]


# ============================================================
# WINDOWS PATH -> WSL PATH
# ============================================================

def windows_to_wsl(path):
    """
    Convert a Windows path on the D: drive to a WSL path.
    Example:
        D:\\folder\\file.avinput
        -> /mnt/d/folder/file.avinput
    """

    path = str(Path(path).resolve())

    match = re.match(r"^([A-Za-z]):[\\/](.*)$", path)

    if not match:
        raise ValueError(
            f"Expected a Windows drive path, received: {path}"
        )

    drive = match.group(1).lower()
    remainder = match.group(2).replace("\\", "/")

    return f"/mnt/{drive}/{remainder}"


# ============================================================
# AMINO ACID POSITION EXTRACTION
# ============================================================

def extract_position_from_name(name):
    """
    Extract the protein residue position and alternate amino acid
    where possible.

    Examples:
        p.Glu295Lys -> (295, K)
        p.T294M     -> (294, M)
        p.Glu295Ter -> (295, None)
        p.Glu295fs  -> (295, None)
    """

    name = str(name)

    # Three-letter amino acid substitution.
    match = re.search(
        r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})",
        name,
    )

    if match:
        position = int(match.group(2))
        alternate = THREE_TO_ONE.get(match.group(3))
        return position, alternate

    # Three-letter amino acid stop variant.
    match = re.search(
        r"p\.([A-Z][a-z]{2})(\d+)Ter",
        name,
    )

    if match:
        return int(match.group(2)), None

    # Three-letter amino acid frameshift.
    match = re.search(
        r"p\.([A-Z][a-z]{2})(\d+)fs",
        name,
    )

    if match:
        return int(match.group(2)), None

    # One-letter amino acid substitution.
    match = re.search(
        r"p\.([A-Z])(\d+)([A-Z])",
        name,
    )

    if match:
        return int(match.group(2)), match.group(3)

    # One-letter stop variant.
    match = re.search(
        r"p\.([A-Z])(\d+)\*",
        name,
    )

    if match:
        return int(match.group(2)), None

    # Approximate codon position from a coding-DNA substitution.
    match = re.search(
        r"c\.(\d+)[ACGT]>[ACGT]",
        name,
    )

    if match:
        nucleotide_position = int(match.group(1))
        return (nucleotide_position + 2) // 3, None

    # Approximate codon position from a coding-DNA indel.
    match = re.search(
        r"c\.(\d+)(?:del|ins|dup)",
        name,
    )

    if match:
        nucleotide_position = int(match.group(1))
        return (nucleotide_position + 2) // 3, None

    return None, None


# ============================================================
# STRUCTURAL FEATURE EXTRACTION
# ============================================================

def run_structural_feature_extraction():
    """
    Extract residue-level structural features from the configured
    PDB file.

    Note:
    B-factor values in an experimental PDB are not necessarily
    AlphaFold pLDDT values. This code retains the existing pLDDT
    column name for compatibility with the pipeline.
    """

    if not PDB_FILE.exists():
        raise FileNotFoundError(
            f"PDB structure not found:\n{PDB_FILE}"
        )

    print("\nExtracting CLN2 structural features...")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("CLN2", str(PDB_FILE))
    model = structure[0]

    residues = []

    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue

            if "CA" not in residue:
                continue

            residues.append(residue)

    if not residues:
        raise ValueError(
            "No standard protein residues were found in the PDB."
        )

    # B-factor values.
    plddt_data = []

    for residue in residues:
        plddt_data.append({
            "residue_number": int(residue.id[1]),
            "amino_acid": residue.resname,
            "pLDDT": float(residue["CA"].get_bfactor()),
        })

    df_plddt = pd.DataFrame(plddt_data)

    # Secondary structure.
    with open(PDB_FILE, "r", encoding="utf-8", errors="ignore") as handle:
        pdb_text = handle.read()

    coordinates = pydssp.read_pdbtext(pdb_text)
    secondary_structure = list(
        pydssp.assign(coordinates, out_type="c3")
    )

    if len(secondary_structure) != len(df_plddt):
        raise ValueError(
            "Secondary-structure length does not match residue count. "
            f"Residues={len(df_plddt)}, "
            f"secondary-structure values={len(secondary_structure)}."
        )

    df_plddt["secondary_structure"] = secondary_structure

    # Relative solvent accessibility.
    sr = ShrakeRupley()
    sr.compute(structure, level="R")

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
        "VAL": 165,
    }

    rsa_data = []

    for residue in residues:
        residue_name = residue.get_resname()
        asa = getattr(residue, "sasa", None)

        if asa is None:
            rsa = None
        else:
            rsa = round(
                float(asa) / max_asa.get(residue_name, 200),
                3,
            )

        rsa_data.append({
            "residue_number": int(residue.id[1]),
            "RSA": rsa,
        })

    df_rsa = pd.DataFrame(rsa_data)

    # Contact count. Count distinct residue numbers within the
    # configured distance of each residue's CA atom.
    atoms = list(structure.get_atoms())
    neighbor_search = NeighborSearch(atoms)

    contact_data = []

    for residue in residues:
        residue_number = int(residue.id[1])
        ca = residue["CA"]

        neighbors = neighbor_search.search(
            ca.get_coord(),
            CONTACT_RADIUS,
        )

        neighboring_residues = set()

        for atom in neighbors:
            neighbor_residue = atom.get_parent()

            if neighbor_residue.id[0] != " ":
                continue

            if int(neighbor_residue.id[1]) == residue_number:
                continue

            neighboring_residues.add(
                (
                    neighbor_residue.get_parent().id,
                    int(neighbor_residue.id[1]),
                )
            )

        contact_data.append({
            "residue_number": residue_number,
            "contact_count": len(neighboring_residues),
        })

    df_contacts = pd.DataFrame(contact_data)

    # Combine structural features.
    result = (
        df_plddt
        .merge(df_rsa, on="residue_number", how="left")
        .merge(df_contacts, on="residue_number", how="left")
    )

    result["burial"] = result["RSA"].apply(
        lambda value:
            "buried" if pd.notna(value) and value < 0.2
            else (
                "exposed" if pd.notna(value) and value > 0.5
                else (
                    "intermediate" if pd.notna(value)
                    else None
                )
            )
    )

    result.to_csv(STRUCTURAL_OUTPUT, index=False)

    print(f"Saved structural features: {STRUCTURAL_OUTPUT}")
    print(f"Structural dataset shape: {result.shape}")

    return result


def load_structural_features():
    """
    Extract and load structural features, then rename the residue
    number column for merging with ClinVar variants.
    """

    run_structural_feature_extraction()

    pdb = pd.read_csv(STRUCTURAL_OUTPUT)

    pdb = pdb.rename(
        columns={"residue_number": "residue_position"}
    )

    pdb["residue_position"] = pd.to_numeric(
        pdb["residue_position"],
        errors="coerce",
    )

    return pdb[
        [
            "residue_position",
            "pLDDT",
            "secondary_structure",
            "RSA",
            "burial",
            "contact_count",
        ]
    ]


# ============================================================
# ALPHAMISSENSE
# ============================================================

def load_alphamissense():
    if not ALPHAMISSENSE_FILE.exists():
        raise FileNotFoundError(
            f"AlphaMissense file not found:\n{ALPHAMISSENSE_FILE}"
        )

    print("\nLoading AlphaMissense data...")

    am = pd.read_csv(
        ALPHAMISSENSE_FILE,
        sep="\t",
        low_memory=False,
    )

    required = [
        "position",
        "a.a.2",
        "pathogenicity score",
        "pathogenicity class",
    ]

    missing = [column for column in required if column not in am.columns]

    if missing:
        raise ValueError(
            "Missing required AlphaMissense columns: "
            + ", ".join(missing)
        )

    am = am[required].copy()

    am.columns = [
        "residue_position",
        "alt_aa",
        "am_pathogenicity",
        "am_class",
    ]

    am["residue_position"] = pd.to_numeric(
        am["residue_position"],
        errors="coerce",
    )

    am["alt_aa"] = am["alt_aa"].astype(str).str.strip()

    am["am_pathogenicity"] = pd.to_numeric(
        am["am_pathogenicity"],
        errors="coerce",
    )

    am = am.drop_duplicates(
        subset=["residue_position", "alt_aa"]
    )

    print(f"AlphaMissense entries: {len(am)}")

    return am


# ============================================================
# CADD-INDEL QUERY
# ============================================================

def query_cadd_indel(chrom, pos, ref, alt):
    """
    Query the configured bgzip/tabix-indexed CADD indel file.

    Expected record layout:
        chromosome, position, reference, alternate, raw score, PHRED

    If the local CADD file uses a different layout, adjust the
    field positions below to match its header.
    """

    chromosome = str(chrom).replace("chr", "")
    region = f"{chromosome}:{pos}-{pos}"

    command = (
        f"tabix {WSL_CADD_FILE} "
        f"{shlex.quote(region)}"
    )

    try:
        result = subprocess.run(
            ["wsl", "bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        if result.returncode != 0:
            return None, None

        for line in result.stdout.splitlines():
            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 6:
                continue

            # Match the exact reference and alternate alleles.
            if parts[2] == ref and parts[3] == alt:
                try:
                    return float(parts[4]), float(parts[5])
                except (ValueError, IndexError):
                    continue

    except (OSError, subprocess.TimeoutExpired):
        pass

    return None, None


def get_cadd_indel_scores(df_indels):
    df_indels = df_indels.copy()

    raw_scores = []
    phred_scores = []

    total = len(df_indels)

    print(f"\nQuerying CADD-Indel for {total} variants...")

    for index, (_, row) in enumerate(df_indels.iterrows(), start=1):
        if index == 1 or index % 50 == 0 or index == total:
            print(f"  CADD progress: {index}/{total}")

        try:
            position = int(row["Start"])
        except (TypeError, ValueError):
            raw_scores.append(None)
            phred_scores.append(None)
            continue

        raw, phred = query_cadd_indel(
            row["Chromosome"],
            position,
            str(row["ReferenceAllele"]),
            str(row["AlternateAllele"]),
        )

        raw_scores.append(raw)
        phred_scores.append(phred)

    df_indels["CADD_indel_raw"] = raw_scores
    df_indels["CADD_indel_phred"] = phred_scores

    matched = df_indels["CADD_indel_phred"].notna().sum()

    print(f"CADD-Indel matched: {matched}/{len(df_indels)}")

    return df_indels


# ============================================================
# ANNOVAR SNP ANNOTATION
# ============================================================

def annotate_snps_annovar(df_snps, output_prefix):
    """
    Run ANNOVAR on SNPs.

    Fixes:
      - Uses $HOME in the WSL shell rather than quoting ~/...
      - Uses separate -protocol and -operation arguments
      - Quotes input/output paths
      - Checks that ANNOVAR files exist
    """

    df_snps = df_snps.copy()

    if df_snps.empty:
        return df_snps

    df_snps["Chromosome"] = (
        df_snps["Chromosome"]
        .astype(str)
        .str.replace("chr", "", regex=False)
    )

    df_snps["Start"] = pd.to_numeric(
        df_snps["Start"],
        errors="coerce",
    )

    df_snps = df_snps.dropna(subset=["Start"]).copy()
    df_snps["Start"] = df_snps["Start"].astype(int)

    input_file = str(output_prefix) + ".avinput"

    annovar_input = df_snps[
        [
            "Chromosome",
            "Start",
            "Start",
            "ReferenceAllele",
            "AlternateAllele",
        ]
    ].copy()

    annovar_input.columns = ["chr", "start", "end", "ref", "alt"]

    annovar_input.to_csv(
        input_file,
        sep="\t",
        header=False,
        index=False,
    )

    wsl_input = windows_to_wsl(input_file)
    wsl_output = windows_to_wsl(str(output_prefix))

    # IMPORTANT:
    # Do not write cd '~/annovar_work/annovar'.
    # Quoting the tilde prevents Bash from expanding it.
    #
    # $HOME is expanded by the shell, and all data paths are quoted.
    command = (
        f'cd "{WSL_ANNOVAR_DIR}" && '
        f'test -f ./table_annovar.pl && '
        f'test -d ./humandb && '
        f'./table_annovar.pl {shlex.quote(wsl_input)} humandb/ '
        f'-buildver hg38 '
        f'-out {shlex.quote(wsl_output)} '
        f'-remove '
        f'-protocol refGene,avsnp150,dbnsfp42a '
        f'-operation g,f,f '
        f'-nastring . '
        f'-csvout '
        f'-otherinfo '
        f"-arg '-splicing 5',,"
    )

    print("\nRunning ANNOVAR...")
    print(f"Input: {input_file}")
    print("ANNOVAR directory: $HOME/annovar_work/annovar")

    subprocess.run(
        ["wsl", "bash", "-lc", command],
        check=True,
    )

    annotated_file = str(output_prefix) + ".hg38_multianno.csv"

    if not os.path.exists(annotated_file):
        raise FileNotFoundError(
            "ANNOVAR completed but its expected output file was not found:\n"
            f"{annotated_file}"
        )

    annotation = pd.read_csv(
        annotated_file,
        low_memory=False,
    )

    # The ANNOVAR output contains annotation columns and the original
    # chromosome/start/end/reference/alternate fields.
    annotation.columns = annotation.columns.str.strip()

    # Normalise chromosome fields before merging.
    if "Chr" in annotation.columns:
        annotation["Chr"] = (
            annotation["Chr"]
            .astype(str)
            .str.replace("chr", "", regex=False)
        )

    if "Start" in annotation.columns:
        annotation["Start"] = pd.to_numeric(
            annotation["Start"],
            errors="coerce",
        )

    df_snps["Start"] = pd.to_numeric(
        df_snps["Start"],
        errors="coerce",
    )

    merge_keys_left = [
        "Chromosome",
        "Start",
        "ReferenceAllele",
        "AlternateAllele",
    ]

    merge_keys_right = ["Chr", "Start", "Ref", "Alt"]

    required_keys = merge_keys_right

    missing_keys = [
        column
        for column in required_keys
        if column not in annotation.columns
    ]

    if missing_keys:
        raise ValueError(
            "Expected ANNOVAR merge columns are missing: "
            + ", ".join(missing_keys)
            + "\nAvailable columns: "
            + ", ".join(annotation.columns.astype(str))
        )

    # Avoid duplicate columns when the input already contains an
    # annotation field with the same name.
    duplicate_columns = [
        column
        for column in annotation.columns
        if column in df_snps.columns
        and column not in merge_keys_right
    ]

    annotation = annotation.drop(
        columns=duplicate_columns,
        errors="ignore",
    )

    df_snps = df_snps.merge(
        annotation,
        left_on=merge_keys_left,
        right_on=merge_keys_right,
        how="left",
        suffixes=("", "_annovar"),
    )

    print(f"SNPs after ANNOVAR: {len(df_snps)}")

    return df_snps


# ============================================================
# LOAD AND FILTER CLINVAR
# ============================================================

def load_clinvar(input_file, mode):
    """
    mode='labelled':
        Keep pathogenic, likely pathogenic, benign and likely benign.

    mode='vus':
        Keep uncertain-significance variants only.
    """

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"ClinVar input file not found:\n{input_file}"
        )

    print(f"\nLoading ClinVar: {input_file}")

    df = pd.read_csv(
        input_file,
        sep="\t",
        low_memory=False,
    )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("#", "", regex=False)
    )

    required = [
        "VariationID",
        "GeneSymbol",
        "ClinicalSignificance",
        "Chromosome",
        "PositionVCF",
        "Assembly",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "Name",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            "Missing required ClinVar columns: "
            + ", ".join(missing)
        )

    df = df[required].copy()

    df = df.rename(
        columns={
            "ReferenceAlleleVCF": "ReferenceAllele",
            "AlternateAlleleVCF": "AlternateAllele",
            "PositionVCF": "Start",
        }
    )

    # Gene filter.
    df = df[
        df["GeneSymbol"].astype(str).str.strip().eq(GENE)
    ].copy()

    # Assembly filter.
    df = df[
        df["Assembly"].astype(str).str.strip().eq("GRCh38")
    ].copy()

    print(f"TPP1 GRCh38 variants: {len(df)}")

    df["ClinicalSignificance"] = (
        df["ClinicalSignificance"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    if mode == "vus":
        vus_terms = {
            "uncertain significance",
            "vus",
        }

        df = df[
            df["ClinicalSignificance"].isin(vus_terms)
        ].copy()

        # VUS have no known supervised label.
        df["label"] = pd.NA

    else:
        label_map = {
            "pathogenic": 1,
            "likely pathogenic": 1,
            "benign": 0,
            "likely benign": 0,
        }

        df = df[
            df["ClinicalSignificance"].isin(label_map)
        ].copy()

        df["label"] = df["ClinicalSignificance"].map(label_map)

    df = df.drop_duplicates(
        subset=["VariationID"],
    ).copy()

    # Clean alleles and chromosome.
    for column in [
        "ReferenceAllele",
        "AlternateAllele",
        "Chromosome",
    ]:
        df[column] = df[column].astype(str).str.strip()

    invalid_alleles = {"", ".", "na", "nan", "none"}

    valid_alleles = (
        ~df["ReferenceAllele"].str.lower().isin(invalid_alleles)
        & ~df["AlternateAllele"].str.lower().isin(invalid_alleles)
        & ~df["Chromosome"].str.lower().isin(invalid_alleles)
    )

    df = df[valid_alleles].copy()

    df["Start"] = pd.to_numeric(
        df["Start"],
        errors="coerce",
    )

    df = df.dropna(subset=["Start"]).copy()
    df["Start"] = df["Start"].astype(int)

    # Extract protein position and alternate amino acid.
    extracted = df["Name"].apply(extract_position_from_name)

    df["residue_position"] = extracted.apply(lambda value: value[0])
    df["alt_aa"] = extracted.apply(lambda value: value[1])

    df["residue_position"] = pd.to_numeric(
        df["residue_position"],
        errors="coerce",
    )

    print(f"Variants retained for mode '{mode}': {len(df)}")

    if mode == "labelled":
        print("Pathogenic:", int((df["label"] == 1).sum()))
        print("Benign:", int((df["label"] == 0).sum()))
    else:
        print("VUS candidates:", len(df))

    return df


# ============================================================
# FEATURE ENGINEERING AND OUTPUT
# ============================================================

def preprocess_dataset(input_file, output_prefix, mode, am, pdb):
    """
    Run annotation, feature merges, missingness filtering and saving
    for one dataset.
    """

    df = load_clinvar(input_file, mode)

    if df.empty:
        print(f"No variants to process for mode '{mode}'.")
        return

    # Split SNPs and indels.
    is_snp = (
        df["ReferenceAllele"].str.len().eq(1)
        & df["AlternateAllele"].str.len().eq(1)
    )

    snps = df[is_snp].copy()
    indels = df[~is_snp].copy()

    print(f"\nSNPs: {len(snps)}")
    print(f"Indels: {len(indels)}")

    # Annotate SNPs using ANNOVAR.
    snps = annotate_snps_annovar(
        snps,
        output_prefix,
    )

    # Query CADD-Indel for indels.
    if not indels.empty:
        indels = get_cadd_indel_scores(indels)

    # Combine SNP and indel records.
    df = pd.concat(
        [snps, indels],
        ignore_index=True,
        sort=False,
    )

    if df.empty:
        print("No variants remain after annotation.")
        return

    # Variant length.
    df["var_length"] = (
        df["AlternateAllele"].astype(str).str.len()
        - df["ReferenceAllele"].astype(str).str.len()
    )

    # Merge AlphaMissense.
    df = df.merge(
        am,
        on=["residue_position", "alt_aa"],
        how="left",
        suffixes=("", "_am"),
    )

    df["am_pathogenicity"] = pd.to_numeric(
        df["am_pathogenicity"],
        errors="coerce",
    )

    df["am_likely_pathogenic"] = (
        df["am_class"].astype(str).str.lower().eq("likely_pathogenic")
    ).astype(int)

    df["am_ambiguous"] = (
        df["am_class"].astype(str).str.lower().eq("ambiguous")
    ).astype(int)

    print(
        "AlphaMissense matched:",
        int(df["am_pathogenicity"].notna().sum()),
        "/",
        len(df),
    )

    # Merge structural features.
    df = df.merge(
        pdb,
        on="residue_position",
        how="left",
        suffixes=("", "_structure"),
    )

    print(
        "Structural records matched:",
        int(df["pLDDT"].notna().sum()),
        "/",
        len(df),
    )

    # Encode structural categories.
    df["ss_helix"] = (
        df["secondary_structure"].astype(str).eq("H")
    ).astype(int)

    df["ss_sheet"] = (
        df["secondary_structure"].astype(str).eq("E")
    ).astype(int)

    df["ss_coil"] = (
        df["secondary_structure"].astype(str).eq("-")
    ).astype(int)

    df["is_buried"] = (
        df["burial"].astype(str).eq("buried")
    ).astype(int)

    df["is_exposed"] = (
        df["burial"].astype(str).eq("exposed")
    ).astype(int)

    df["is_intermediate"] = (
        df["burial"].astype(str).eq("intermediate")
    ).astype(int)

    # Ensure every expected feature exists.
    for feature in FEATURES:
        if feature not in df.columns:
            df[feature] = pd.NA

    # Convert features to numeric.
    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature].replace(".", pd.NA),
            errors="coerce",
        )

    # Remove rows based on feature coverage.
    # Only feature columns are counted, not identifiers or labels.
    feature_count = len(FEATURES)

    min_snp_features = max(
        1,
        int(feature_count * SNP_MIN_FEATURE_COVERAGE),
    )

    min_indel_features = max(
        1,
        int(feature_count * INDEL_MIN_FEATURE_COVERAGE),
    )

    is_snp_row = df["var_length"].eq(0)

    snp_rows = df[is_snp_row].copy()
    indel_rows = df[~is_snp_row].copy()

    snp_rows = snp_rows[
        snp_rows[FEATURES].notna().sum(axis=1) >= min_snp_features
    ].copy()

    indel_rows = indel_rows[
        indel_rows[FEATURES].notna().sum(axis=1) >= min_indel_features
    ].copy()

    df = pd.concat(
        [snp_rows, indel_rows],
        ignore_index=True,
        sort=False,
    )

    # Keep variant identifiers in the VUS file so that predictions
    # can be traced back to the original ClinVar records.
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
        "alt_aa",
    ]

    identifiers = [
        column for column in identifiers
        if column in df.columns
    ]

    if mode == "vus":
        output_df = df[identifiers + FEATURES + ["label"]].copy()
    else:
        output_df = df[FEATURES + ["label"]].copy()

    output_file = str(output_prefix) + "_CLN2_ML_READY.csv"

    output_df.to_csv(
        output_file,
        index=False,
    )

    print("\nPreprocessing complete.")
    print(f"Mode: {mode}")
    print(f"Output: {output_file}")
    print(f"Final shape: {output_df.shape}")

    if mode == "labelled":
        print("Pathogenic rows:", int((output_df["label"] == 1).sum()))
        print("Benign rows:", int((output_df["label"] == 0).sum()))
    else:
        print("VUS candidates retained:", len(output_df))


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("CLN2 / TPP1 PREPROCESSING - CLINVAR 2026")
    print("=" * 70)

    if not CLINVAR_2026.exists():
        raise FileNotFoundError(
            f"2026 ClinVar file not found:\n{CLINVAR_2026}"
        )

    # Check WSL ANNOVAR installation before starting the pipeline.
    check_annovar = (
        f'test -f "{WSL_ANNOVAR_DIR}/table_annovar.pl" && '
        f'test -d "{WSL_ANNOVAR_DIR}/humandb"'
    )

    result = subprocess.run(
        ["wsl", "bash", "-lc", check_annovar],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "ANNOVAR was not found at:\n"
            "$HOME/annovar_work/annovar\n\n"
            "Check the real WSL installation path using:\n"
            'wsl bash -lc \'find "$HOME" -name table_annovar.pl -type f 2>/dev/null | head\''
        )

    # Load annotation resources once.
    am = load_alphamissense()
    pdb = load_structural_features()

    # Labelled dataset: 2026 ClinVar only.
    print("\n" + "=" * 70)
    print("PROCESSING LABELLED 2026 CLN2 DATA")
    print("=" * 70)

    preprocess_dataset(
        input_file=str(CLINVAR_2026),
        output_prefix=str(OUTPUT_DIR / "CLN2_2026_LABELLED"),
        mode="labelled",
        am=am,
        pdb=pdb,
    )

    # VUS dataset: same 2026 ClinVar snapshot.
    # VUS labels remain missing; these are not training labels.
    print("\n" + "=" * 70)
    print("PROCESSING 2026 CLN2 VUS CANDIDATES")
    print("=" * 70)

    preprocess_dataset(
        input_file=str(CLINVAR_2026),
        output_prefix=str(OUTPUT_DIR / "CLN2_VUS"),
        mode="vus",
        am=am,
        pdb=pdb,
    )

    print("\n" + "=" * 70)
    print("CLN2 PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()