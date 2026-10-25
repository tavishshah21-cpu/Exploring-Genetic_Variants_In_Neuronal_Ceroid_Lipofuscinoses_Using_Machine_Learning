# ============================================================
# CLN3 ALPHAFOLD ABLATION PREPROCESSING
# ============================================================
# Creates, for BOTH ClinVar 2025 and 2026:
#
#   CLN3_<YEAR>_WITHOUT_AF_CLN3_ML_READY.csv
#   CLN3_<YEAR>_WITH_AF_CLN3_ML_READY.csv
#
# FAIR ABLATION DESIGN:
#
# WITHOUT AF:
#   Biological / evolutionary / prediction features
#
# WITH AF:
#   Same biological features
#   +
#   AlphaFold structural features
#
# IMPORTANT:
# - Both datasets contain EXACTLY the same variants.
# - Filtering is based on the common baseline features.
# - AlphaFold does NOT decide which variants are retained.
# - ClinVar 2025 and 2026 are processed independently.
# - ML training/testing is handled by the separate ML script.
# ============================================================

import re
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = Path(
    r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics"
)

GENE = "CLN3"

WSL_ANNOVAR_DIR = "~/annovar_work/annovar"

CADD_INDEL_FILE = (
    "~/cadd_data/gnomad.genomes.r4.0.indel.tsv.gz"
)

INPUTS = {
    2025: SOURCE_DIR / "variant_summary_2025.txt",
    2026: SOURCE_DIR / "variant_summary_2026.txt",
}

AM_FILE = (
    SOURCE_DIR /
    "AlphaMissense-Search-Q13286.tsv"
)

PDB_FILE = (
    SOURCE_DIR /
    "cln3_structural_features.csv"
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
    "Val": "V",
}


# ============================================================
# BIOLOGICAL FEATURES
# ============================================================

BIOLOGICAL_FEATURES = [

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
]


# ============================================================
# ALPHAFOLD FEATURES
# ============================================================

STRUCTURAL_FEATURES = [

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
# PRINT SECTION
# ============================================================

def print_section(title):

    print("\n" + "=" * 70)

    print(title)

    print("=" * 70)


# ============================================================
# EXTRACT PROTEIN POSITION
# ============================================================

def extract_position_from_name(name):

    """
    Extract residue position and alternate amino acid
    from the ClinVar Name column.

    Handles:

        p.Glu295Lys
        p.Tyr199Ter
        p.Leu350fs
        p.T294M
        p.T294*
        c.1000C>T
        c.1048del
        c.1048ins
        c.1048dup
    """

    name = str(name)

    # --------------------------------------------------------
    # Three-letter missense
    # Example: p.Glu295Lys
    # --------------------------------------------------------

    match = re.search(
        r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})",
        name
    )

    if match:

        position = int(match.group(2))

        alt = THREE_TO_ONE.get(
            match.group(3)
        )

        return position, alt


    # --------------------------------------------------------
    # Three-letter stop
    # Example: p.Tyr199Ter
    # --------------------------------------------------------

    match = re.search(
        r"p\.([A-Z][a-z]{2})(\d+)Ter",
        name
    )

    if match:

        position = int(match.group(2))

        return position, None


    # --------------------------------------------------------
    # Frameshift
    # Example: p.Leu350fs
    # --------------------------------------------------------

    match = re.search(
        r"p\.([A-Z][a-z]{2})(\d+)fs",
        name
    )

    if match:

        position = int(match.group(2))

        return position, None


    # --------------------------------------------------------
    # Single-letter missense
    # Example: p.T294M
    # --------------------------------------------------------

    match = re.search(
        r"p\.([A-Z])(\d+)([A-Z])",
        name
    )

    if match:

        position = int(match.group(2))

        alt = match.group(3)

        return position, alt


    # --------------------------------------------------------
    # Single-letter stop
    # Example: p.T294*
    # --------------------------------------------------------

    match = re.search(
        r"p\.([A-Z])(\d+)\*",
        name
    )

    if match:

        position = int(match.group(2))

        return position, None


    # --------------------------------------------------------
    # Coding substitution
    # Example: c.1000C>T
    #
    # Approximate AA position:
    # ceil(nt_position / 3)
    # --------------------------------------------------------

    match = re.search(
        r"c\.(\d+)[ACGT]>[ACGT]",
        name
    )

    if match:

        nt_position = int(
            match.group(1)
        )

        aa_position = (
            nt_position + 2
        ) // 3

        return aa_position, None


    # --------------------------------------------------------
    # Coding deletion / insertion / duplication
    # --------------------------------------------------------

    match = re.search(
        r"c\.(\d+)(?:del|ins|dup)",
        name
    )

    if match:

        nt_position = int(
            match.group(1)
        )

        aa_position = (
            nt_position + 2
        ) // 3

        return aa_position, None


    return None, None


# ============================================================
# LOAD ALPHAMISSENSE
# ============================================================

def load_alphamissense():

    print_section(
        "LOADING ALPHAMISSENSE"
    )

    if not AM_FILE.exists():

        raise FileNotFoundError(
            f"AlphaMissense file not found:\n{AM_FILE}"
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

        "pathogenicity class",

    ]


    missing = [

        column

        for column in required

        if column not in am.columns

    ]


    if missing:

        raise ValueError(
            f"AlphaMissense missing columns: "
            f"{missing}"
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

        errors="coerce"

    )


    am["alt_aa"] = (

        am["alt_aa"]

        .astype(str)

        .str.strip()

        .str.upper()

    )


    am["am_pathogenicity"] = pd.to_numeric(

        am["am_pathogenicity"],

        errors="coerce"

    )


    am["am_class"] = (

        am["am_class"]

        .astype(str)

        .str.strip()

        .str.lower()

        .str.replace(
            " ",
            "_",
            regex=False
        )

    )


    am = am[

        am["residue_position"].notna()

        & (am["alt_aa"] != "")

        & (am["alt_aa"] != "NAN")

    ].copy()


    am = am.drop_duplicates(

        subset=[
            "residue_position",
            "alt_aa"
        ]

    )


    print(
        f"AlphaMissense entries: {len(am)}"
    )


    return am


# ============================================================
# LOAD ALPHAFOLD STRUCTURAL FEATURES
# ============================================================

def load_structural_features():

    print_section(
        "LOADING ALPHAFOLD STRUCTURAL FEATURES"
    )

    if not PDB_FILE.exists():

        raise FileNotFoundError(
            f"Structural feature file not found:\n"
            f"{PDB_FILE}"
        )


    pdb = pd.read_csv(
        PDB_FILE,
        low_memory=False
    )


    required = [

        "residue_number",

        "pLDDT",

        "secondary_structure",

        "RSA",

        "burial",

    ]


    missing = [

        column

        for column in required

        if column not in pdb.columns

    ]


    if missing:

        raise ValueError(
            f"Structural file missing columns: "
            f"{missing}"
        )


    columns = required.copy()


    if "contact_count" in pdb.columns:

        columns.append(
            "contact_count"
        )

        print(
            "contact_count found."
        )

    else:

        print(
            "WARNING: contact_count not found."
        )


    pdb = pdb[columns].copy()


    pdb = pdb.rename(

        columns={
            "residue_number":
            "residue_position"
        }

    )


    pdb["residue_position"] = pd.to_numeric(

        pdb["residue_position"],

        errors="coerce"

    )


    # --------------------------------------------------------
    # Secondary structure
    # --------------------------------------------------------

    ss = (

        pdb["secondary_structure"]

        .astype(str)

        .str.upper()

        .str.strip()

    )


    pdb["ss_helix"] = (

        ss == "H"

    ).astype(int)


    pdb["ss_sheet"] = (

        ss == "E"

    ).astype(int)


    pdb["ss_coil"] = (

        ss == "-"

    ).astype(int)


    # --------------------------------------------------------
    # Burial
    # --------------------------------------------------------

    burial = (

        pdb["burial"]

        .astype(str)

        .str.lower()

        .str.strip()

    )


    pdb["is_buried"] = (

        burial == "buried"

    ).astype(int)


    pdb["is_exposed"] = (

        burial == "exposed"

    ).astype(int)


    pdb["is_intermediate"] = (

        burial == "intermediate"

    ).astype(int)


    print(
        f"Structural entries: {len(pdb)}"
    )


    return pdb


# ============================================================
# QUERY CADD-INDEL
# ============================================================

def query_cadd_indel(
    chrom,
    pos,
    ref,
    alt
):

    try:

        region = (
            f"{chrom}:{pos}-{pos}"
        )


        command = [

            "wsl",

            "bash",

            "-c",

            f"tabix "
            f"{CADD_INDEL_FILE} "
            f"{region}",

        ]


        result = subprocess.run(

            command,

            capture_output=True,

            text=True,

            timeout=10

        )


        output = (
            result.stdout
            .strip()
        )


        if not output:

            return None, None


        for line in output.splitlines():

            parts = line.split("\t")


            if len(parts) < 6:

                continue


            if (

                parts[2] == ref

                and

                parts[3] == alt

            ):

                raw = float(
                    parts[4]
                )

                phred = float(
                    parts[5]
                )

                return raw, phred


    except Exception:

        pass


    return None, None


# ============================================================
# ADD CADD-INDEL SCORES
# ============================================================

def add_cadd_indel_scores(df):

    df = df.copy()


    raw_scores = []

    phred_scores = []


    for i, (_, row) in enumerate(
        df.iterrows()
    ):

        if i % 50 == 0:

            print(
                f"  CADD-Indel: "
                f"{i}/{len(df)}"
            )


        chrom = (

            str(row["Chromosome"])

            .replace("chr", "")

        )


        try:

            pos = int(
                row["Start"]
            )

        except Exception:

            raw_scores.append(None)

            phred_scores.append(None)

            continue


        ref = str(
            row["ReferenceAllele"]
        )

        alt = str(
            row["AlternateAllele"]
        )


        raw, phred = query_cadd_indel(

            chrom,

            pos,

            ref,

            alt

        )


        raw_scores.append(raw)

        phred_scores.append(phred)


    df["CADD_indel_raw"] = (
        raw_scores
    )

    df["CADD_indel_phred"] = (
        phred_scores
    )


    matched = sum(

        x is not None

        for x in phred_scores

    )


    print(

        f"CADD-Indel matched: "
        f"{matched} / {len(df)}"

    )


    return df


# ============================================================
# LOAD + FILTER CLINVAR
# ============================================================

def load_and_filter(input_file):

    print_section(
        f"LOADING {input_file.name}"
    )


    if not input_file.exists():

        raise FileNotFoundError(
            f"ClinVar input not found:\n"
            f"{input_file}"
        )


    df = pd.read_csv(

        input_file,

        sep="\t",

        low_memory=False

    )


    print(
        f"Original rows: {len(df)}"
    )


    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = (

        df.columns

        .str.strip()

        .str.replace(
            "#",
            "",
            regex=False
        )

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


    missing = [

        c

        for c in required

        if c not in df.columns

    ]


    if missing:

        raise ValueError(

            f"{input_file.name} "
            f"missing columns: {missing}"

        )


    df = df[required].copy()


    # --------------------------------------------------------
    # Rename
    # --------------------------------------------------------

    df = df.rename(

        columns={

            "ReferenceAlleleVCF":
            "ReferenceAllele",

            "AlternateAlleleVCF":
            "AlternateAllele",

            "PositionVCF":
            "Start",

        }

    )


    # --------------------------------------------------------
    # CLN3
    # --------------------------------------------------------

    df = df[

        df["GeneSymbol"]

        .astype(str)

        .str.upper()

        == GENE

    ].copy()


    print(
        f"CLN3 variants: {len(df)}"
    )


    # --------------------------------------------------------
    # GRCh38
    # --------------------------------------------------------

    df = df[

        df["Assembly"]

        .astype(str)

        .str.upper()

        == "GRCH38"

    ].copy()


    print(
        f"After GRCh38: {len(df)}"
    )


    # --------------------------------------------------------
    # Clinical significance
    # --------------------------------------------------------

    df["ClinicalSignificance"] = (

        df["ClinicalSignificance"]

        .astype(str)

        .str.strip()

        .str.lower()

    )


    label_map = {

        "pathogenic": 1,

        "likely pathogenic": 1,

        "benign": 0,

        "likely benign": 0,

    }


    df = df[

        df["ClinicalSignificance"]

        .isin(label_map.keys())

    ].copy()


    df["label"] = (

        df["ClinicalSignificance"]

        .map(label_map)

    )


    # --------------------------------------------------------
    # Remove duplicate VariationIDs
    # --------------------------------------------------------

    before = len(df)


    df = df.drop_duplicates(

        subset="VariationID"

    ).copy()


    print(
        f"Duplicates removed: "
        f"{before - len(df)}"
    )


    # --------------------------------------------------------
    # Extract protein position + alt AA
    # --------------------------------------------------------

    extracted = (

        df["Name"]

        .apply(
            extract_position_from_name
        )

    )


    df["residue_position"] = (

        extracted.apply(
            lambda x: x[0]
        )

    )


    df["alt_aa"] = (

        extracted.apply(
            lambda x: x[1]
        )

    )


    # --------------------------------------------------------
    # Numeric position
    # --------------------------------------------------------

    df["Start"] = pd.to_numeric(

        df["Start"],

        errors="coerce"

    )


    # --------------------------------------------------------
    # Clean alleles
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


    valid = (

        df["ReferenceAllele"].notna()

        & df["AlternateAllele"].notna()

        & (

            df["ReferenceAllele"]

            .str.lower()

            != "na"

        )

        & (

            df["AlternateAllele"]

            .str.lower()

            != "na"

        )

        & (

            df["ReferenceAllele"]

            != "."

        )

        & (

            df["AlternateAllele"]

            != "."

        )

        & (

            df["ReferenceAllele"]

            != ""

        )

        & (

            df["AlternateAllele"]

            != ""

        )

        & df["Chromosome"].notna()

        & df["Start"].notna()

        & (df["Start"] > 0)

    )


    df = df[valid].copy()


    df = df[

        ~df["Chromosome"]

        .astype(str)

        .str.contains(
            "na",
            case=False
        )

    ].copy()


    print(
        f"After invalid removal: "
        f"{len(df)}"
    )


    return df


# ============================================================
# PROCESS SNPs THROUGH ANNOVAR
# ============================================================

def annotate_snps(
    df_snps,
    output_prefix
):

    if len(df_snps) == 0:

        return df_snps.copy()


    df_snps = df_snps.copy()


    # --------------------------------------------------------
    # Ensure chr prefix
    # --------------------------------------------------------

    df_snps["Chromosome"] = (

        df_snps["Chromosome"]

        .astype(str)

        .apply(

            lambda x:

            x

            if x.startswith("chr")

            else "chr" + x

        )

    )


    annovar_input = (

        str(output_prefix)

        + ".avinput"

    )


    annovar_df = df_snps[

        [

            "Chromosome",

            "Start",

            "Start",

            "ReferenceAllele",

            "AlternateAllele",

        ]

    ].copy()


    annovar_df.columns = [

        "chr",

        "start",

        "end",

        "ref",

        "alt",

    ]


    annovar_df.to_csv(

        annovar_input,

        sep="\t",

        header=False,

        index=False

    )


    print(
        f"ANNOVAR input: "
        f"{annovar_input}"
    )


    # --------------------------------------------------------
    # Windows -> WSL
    # --------------------------------------------------------

    wsl_input = (

        annovar_input

        .replace(
            "D:",
            "/mnt/d"
        )

        .replace(
            "\\",
            "/"
        )

    )


    wsl_output = (

        str(output_prefix)

        .replace(
            "D:",
            "/mnt/d"
        )

        .replace(
            "\\",
            "/"
        )

    )


    # --------------------------------------------------------
    # ANNOVAR command
    # --------------------------------------------------------

    command = [

        "wsl",

        "bash",

        "-c",

        (

            f"cd {WSL_ANNOVAR_DIR} && "

            f"./table_annovar.pl "

            f"{wsl_input} "

            f"humandb/ "

            f"-buildver hg38 "

            f"-out {wsl_output} "

            f"-remove "

            f"-protocol "
            f"refGene,avsnp150,dbnsfp42a "

            f"-operation g,f,f "

            f"-nastring . "

            f"-csvout "

            f"-otherinfo "

            f"-arg '-splicing 5',,"

        )

    ]


    print(
        "\nRunning ANNOVAR..."
    )


    subprocess.run(
        command,
        check=True
    )


    ann_file = Path(

        str(output_prefix)

        + ".hg38_multianno.csv"

    )


    if not ann_file.exists():

        raise FileNotFoundError(

            f"ANNOVAR output not found:\n"
            f"{ann_file}"

        )


    ann = pd.read_csv(

        ann_file,

        low_memory=False

    )


    df_snps["Start"] = pd.to_numeric(

        df_snps["Start"],

        errors="coerce"

    )


    ann["Start"] = pd.to_numeric(

        ann["Start"],

        errors="coerce"

    )


    # --------------------------------------------------------
    # Merge annotations
    # --------------------------------------------------------

    df_snps = df_snps.merge(

        ann,

        left_on=[

            "Chromosome",

            "Start",

            "ReferenceAllele",

            "AlternateAllele",

        ],

        right_on=[

            "Chr",

            "Start",

            "Ref",

            "Alt",

        ],

        how="left",

        suffixes=("", "_ann")

    )


    print(
        f"SNPs after ANNOVAR: "
        f"{len(df_snps)}"
    )


    return df_snps


# ============================================================
# CONSTRUCT WITH / WITHOUT ALPHAFOLD DATASETS
# ============================================================

def construct_datasets(
    df,
    alphamissense,
    structural
):

    print_section(
        "CONSTRUCTING ABLATION DATASETS"
    )


    df = df.copy()


    # --------------------------------------------------------
    # Variant length
    # --------------------------------------------------------

    df["var_length"] = (

        df["AlternateAllele"].str.len()

        -

        df["ReferenceAllele"].str.len()

    )


    # --------------------------------------------------------
    # AlphaMissense
    # --------------------------------------------------------

    df = df.merge(

        alphamissense,

        on=[

            "residue_position",

            "alt_aa",

        ],

        how="left"

    )


    df["am_likely_pathogenic"] = (

        df["am_class"]

        == "likely_pathogenic"

    ).astype(int)


    df["am_ambiguous"] = (

        df["am_class"]

        == "ambiguous"

    ).astype(int)


    # --------------------------------------------------------
    # AlphaFold
    # --------------------------------------------------------

    df = df.merge(

        structural,

        on="residue_position",

        how="left",

        suffixes=("", "_struct")

    )


    # --------------------------------------------------------
    # Recalculate structural encodings
    # --------------------------------------------------------

    if "secondary_structure" in df.columns:

        ss = (

            df["secondary_structure"]

            .astype(str)

            .str.upper()

            .str.strip()

        )


        df["ss_helix"] = (

            ss == "H"

        ).astype(int)


        df["ss_sheet"] = (

            ss == "E"

        ).astype(int)


        df["ss_coil"] = (

            ss == "-"

        ).astype(int)


    if "burial" in df.columns:

        burial = (

            df["burial"]

            .astype(str)

            .str.lower()

            .str.strip()

        )


        df["is_buried"] = (

            burial == "buried"

        ).astype(int)


        df["is_exposed"] = (

            burial == "exposed"

        ).astype(int)


        df["is_intermediate"] = (

            burial == "intermediate"

        ).astype(int)


    # --------------------------------------------------------
    # Detect available features
    # --------------------------------------------------------

    biological_features = [

        f

        for f in BIOLOGICAL_FEATURES

        if f in df.columns

    ]


    structural_features = [

        f

        for f in STRUCTURAL_FEATURES

        if f in df.columns

    ]


    # --------------------------------------------------------
    # WITHOUT AlphaFold
    # --------------------------------------------------------

    without_af_features = list(

        dict.fromkeys(

            biological_features

        )

    )


    # --------------------------------------------------------
    # WITH AlphaFold
    # --------------------------------------------------------

    with_af_features = list(

        dict.fromkeys(

            biological_features

            +

            structural_features

        )

    )


    print(
        "\nWITHOUT AlphaFold features:"
    )

    for feature in without_af_features:

        print(
            f"  BIO: {feature}"
        )


    print(
        "\nWITH AlphaFold additional features:"
    )

    for feature in structural_features:

        print(
            f"  AF:  {feature}"
        )


    print(
        f"\nWITHOUT AF feature count: "
        f"{len(without_af_features)}"
    )


    print(
        f"WITH AF feature count: "
        f"{len(with_af_features)}"
    )


    # ========================================================
    # COMMON BASELINE FILTER
    # ========================================================
    #
    # This is extremely important.
    #
    # The WITHOUT-AF feature set determines whether a variant
    # contains enough information to be retained.
    #
    # Therefore adding AlphaFold features CANNOT cause extra
    # variants to enter the WITH-AF dataset.
    #
    # Both experiments therefore use exactly the same variants.
    # ========================================================

    all_features = list(

        dict.fromkeys(

            with_af_features

            +

            without_af_features

        )

    )


    temp = df[

        [

            "VariationID",

            "label",

        ]

        +

        all_features

    ].copy()


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    temp = temp.replace(
        ".",
        pd.NA
    )


    for feature in all_features:

        temp[feature] = pd.to_numeric(

            temp[feature],

            errors="coerce"

        )


    # --------------------------------------------------------
    # Count baseline feature availability
    # --------------------------------------------------------

    baseline_available = (

        temp[without_af_features]

        .notna()

        .sum(axis=1)

    )


    # --------------------------------------------------------
    # SNP / indel
    # --------------------------------------------------------

    is_snp = (

        temp["var_length"] == 0

    )


    is_indel = (

        temp["var_length"] != 0

    )


    # --------------------------------------------------------
    # Same threshold logic
    #
    # SNP  = 50%
    # Indel = 30%
    # --------------------------------------------------------

    snp_threshold = max(

        1,

        int(

            len(without_af_features)

            * 0.50

        )

    )


    indel_threshold = max(

        1,

        int(

            len(without_af_features)

            * 0.30

        )

    )


    print(
        f"\nBaseline SNP threshold: "
        f"{snp_threshold}"
    )


    print(
        f"Baseline indel threshold: "
        f"{indel_threshold}"
    )


    keep = (

        (

            is_snp

            &

            (

                baseline_available

                >= snp_threshold

            )

        )

        |

        (

            is_indel

            &

            (

                baseline_available

                >= indel_threshold

            )

        )

    )


    temp = temp[keep].copy()


    print(
        f"\nVariants retained for BOTH models: "
        f"{len(temp)}"
    )


    # ========================================================
    # BUILD WITHOUT-AF DATASET
    # ========================================================

    without_af = temp[

        [

            "VariationID"

        ]

        +

        without_af_features

        +

        [

            "label"

        ]

    ].copy()


    # ========================================================
    # BUILD WITH-AF DATASET
    # ========================================================

    with_af = temp[

        [

            "VariationID"

        ]

        +

        with_af_features

        +

        [

            "label"

        ]

    ].copy()


    # ========================================================
    # VERIFY SAME VARIANTS
    # ========================================================

    ids_without = set(

        without_af["VariationID"]

    )


    ids_with = set(

        with_af["VariationID"]

    )


    if ids_without != ids_with:

        raise RuntimeError(

            "WITH-AF and WITHOUT-AF "
            "do not contain the same "
            "VariationIDs."

        )


    if len(without_af) != len(with_af):

        raise RuntimeError(

            "WITH-AF and WITHOUT-AF "
            "have different row counts."

        )


    # --------------------------------------------------------
    # Verify labels
    # --------------------------------------------------------

    labels_without = (

        without_af

        .set_index("VariationID")

        ["label"]

        .sort_index()

    )


    labels_with = (

        with_af

        .set_index("VariationID")

        ["label"]

        .sort_index()

    )


    if not labels_without.equals(
        labels_with
    ):

        raise RuntimeError(

            "Labels differ between "
            "WITH-AF and WITHOUT-AF."

        )


    print(
        "\n✓ Same variants"
    )


    print(
        "✓ Same labels"
    )


    return (
        without_af,
        with_af
    )


# ============================================================
# RUN ONE YEAR
# ============================================================

def run_year(
    year,
    input_file,
    output_prefix,
    alphamissense,
    structural
):

    print_section(
        f"PROCESSING CLN3 {year}"
    )


    # --------------------------------------------------------
    # ClinVar filtering
    # --------------------------------------------------------

    df = load_and_filter(
        input_file
    )


    print(
        f"\nPathogenic: "
        f"{(df['label'] == 1).sum()}"
    )


    print(
        f"Benign: "
        f"{(df['label'] == 0).sum()}"
    )


    # --------------------------------------------------------
    # SNP / indel split
    # --------------------------------------------------------

    is_snp = (

        (

            df["ReferenceAllele"]

            .str.len()

            == 1

        )

        &

        (

            df["AlternateAllele"]

            .str.len()

            == 1

        )

    )


    df_snps = df[
        is_snp
    ].copy()


    df_indels = df[
        ~is_snp
    ].copy()


    print(
        f"\nSNPs: {len(df_snps)}"
    )


    print(
        f"Indels: {len(df_indels)}"
    )


    # --------------------------------------------------------
    # ANNOVAR
    # --------------------------------------------------------

    df_snps = annotate_snps(

        df_snps,

        output_prefix

    )


    # --------------------------------------------------------
    # CADD-Indel
    # --------------------------------------------------------

    df_indels = add_cadd_indel_scores(

        df_indels

    )


    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = pd.concat(

        [

            df_snps,

            df_indels,

        ],

        ignore_index=True,

        sort=False

    )


    print(
        f"\nCombined variants: "
        f"{len(combined)}"
    )


    # --------------------------------------------------------
    # Construct datasets
    # --------------------------------------------------------

    without_af, with_af = (

        construct_datasets(

            combined,

            alphamissense,

            structural

        )

    )


    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    without_path = Path(

        str(output_prefix)

        +

        "_WITHOUT_AF_CLN3_ML_READY.csv"

    )


    with_path = Path(

        str(output_prefix)

        +

        "_WITH_AF_CLN3_ML_READY.csv"

    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    without_af.to_csv(

        without_path,

        index=False

    )


    with_af.to_csv(

        with_path,

        index=False

    )


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_section(
        f"CLN3 {year} OUTPUT"
    )


    print(
        f"WITHOUT AF:\n"
        f"{without_path}"
    )


    print(
        f"\nWITH AF:\n"
        f"{with_path}"
    )


    print(
        f"\nWITHOUT AF shape: "
        f"{without_af.shape}"
    )


    print(
        f"WITH AF shape: "
        f"{with_af.shape}"
    )


    print(
        f"\nSame variants: "
        f"{set(without_af['VariationID']) == set(with_af['VariationID'])}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "CLN3 ALPHAFOLD ABLATION PREPROCESSING"
    )


    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print(
        "\nChecking required files..."
    )


    required_files = [

        INPUTS[2025],

        INPUTS[2026],

        AM_FILE,

        PDB_FILE,

    ]


    for file in required_files:

        if not file.exists():

            raise FileNotFoundError(

                f"Required file not found:\n"
                f"{file}"

            )

        print(
            f"✓ {file}"
        )


    # --------------------------------------------------------
    # Load shared resources
    # --------------------------------------------------------

    alphamissense = (

        load_alphamissense()

    )


    structural = (

        load_structural_features()

    )


    # --------------------------------------------------------
    # Process 2025
    # --------------------------------------------------------

    run_year(

        2025,

        INPUTS[2025],

        BASE_DIR / "CLN3_2025",

        alphamissense,

        structural

    )


    # --------------------------------------------------------
    # Process 2026
    # --------------------------------------------------------

    run_year(

        2026,

        INPUTS[2026],

        BASE_DIR / "CLN3_2026",

        alphamissense,

        structural

    )


    print_section(
        "PREPROCESSING COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()