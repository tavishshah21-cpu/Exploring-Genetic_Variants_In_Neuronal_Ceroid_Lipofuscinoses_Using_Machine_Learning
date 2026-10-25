# ============================================================
# CLN2 INTEGRATED PREPROCESSING PIPELINE
# ============================================================
#
# Based on Preprocessingv3.py
#
# CLN2-specific additions:
#   - TPP1 / CLN2 filtering
#   - CLN2 AlphaMissense
#   - CLN2 structural features
#   - CLN2 output directory
#
# ClinVar source:
#   Pipeline_1_Statistics
#
# CLN2 files:
#   Pipeline_1_cln2
#
# Pipeline:
#
#   ClinVar variant_summary
#           |
#           v
#       Filter TPP1
#           |
#           v
#       Filter GRCh38
#           |
#           v
#       Clinical labels
#           |
#           v
#       SNP / Indel split
#           |
#           +--------------------+
#           |                    |
#           v                    v
#       ANNOVAR              CADD-Indel
#       dbNSFP
#           |                    |
#           +---------+----------+
#                     |
#                     v
#              AlphaMissense
#                     |
#                     v
#             Structural features
#                     |
#                     v
#              Feature engineering
#                     |
#                     v
#             Missing-value filtering
#                     |
#                     v
#              ML-ready CSV
#
# ============================================================


import os
import re
import subprocess

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

GENE = "TPP1"

BASE_DIR = (
    r"D:\tavish_shah\PES\capstone\Pipeline_1_cln2"
)


CLINVAR_DIR = (
    r"D:\tavish_shah\PES\capstone"
    r"\Pipeline_1_Statistics"
)


# ============================================================
# ALPHAMISSENSE
# ============================================================

AM_FILE = (
    r"D:\tavish_shah\PES\capstone"
    r"\Pipeline_1_cln2"
    r"\AlphaMissense-Search-O14773.tsv"
)


# ============================================================
# STRUCTURAL FILE
# ============================================================
#
# This is the PDB generated from the supplied 3EDY CIF.
#
# NOTE:
# The B-factor values in an experimental PDB are not
# technically AlphaFold pLDDT values. The column is retained
# as pLDDT for compatibility with the existing pipeline.
#
# ============================================================

PDB_FILE = (
    r"D:\tavish_shah\PES\capstone"
    r"\Pipeline_1_cln2"
    r"\AF-O14773-F1-model_v6.pdb"
)


PDB_OUTPUT_FILE = (
    r"D:\tavish_shah\PES\capstone"
    r"\Pipeline_1_cln2"
    r"\cln2_structural_features.csv"
)


# ============================================================
# ANNOVAR
# ============================================================

WSL_ANNOVAR_DIR = (
    "~/annovar_work/annovar"
)


# ============================================================
# CADD-INDEL
# ============================================================

CADD_INDEL_FILE = (
    "~/cadd_data/"
    "gnomad.genomes.r4.0.indel.tsv.gz"
)


CONTACT_RADIUS = 8.0


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    BASE_DIR,
    exist_ok=True
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
# STRUCTURAL FEATURE EXTRACTION
# ============================================================

def run_structural_feature_extraction():

    print("\n")
    print("=" * 70)
    print("CLN2 STRUCTURAL FEATURE EXTRACTION")
    print("=" * 70)


    # ========================================================
    # CHECK PDB
    # ========================================================

    if not os.path.exists(PDB_FILE):

        raise FileNotFoundError(

            "\nCLN2 PDB file not found:\n"
            f"{PDB_FILE}\n\n"
            "Place the PDB file in Pipeline_1_cln2."

        )


    # ========================================================
    # LOAD PDB
    # ========================================================

    print(
        "\nLoading CLN2 PDB structure..."
    )


    parser = PDBParser(
        QUIET=True
    )


    structure = parser.get_structure(
        "CLN2",
        PDB_FILE
    )


    model = structure[0]


    # ========================================================
    # EXTRACT pLDDT / B-FACTOR
    # ========================================================

    plddt_data = []


    for chain in model:

        for residue in chain:

            # Keep only standard amino-acid residues.
            #
            # This prevents calcium / ligands / water from
            # being counted as protein residues.

            if residue.id[0] != " ":

                continue


            if not all(

                atom in residue

                for atom in (
                    "N",
                    "CA",
                    "C",
                    "O"
                )

            ):

                continue


            residue_number = (
                residue.id[1]
            )


            amino_acid = (
                residue.resname
            )


            plddt = (
                residue["CA"]
                .get_bfactor()
            )


            plddt_data.append({

                "residue_number":
                    residue_number,

                "amino_acid":
                    amino_acid,

                "pLDDT":
                    plddt

            })


    df_plddt = pd.DataFrame(
        plddt_data
    )


    print(
        f"Residues extracted: "
        f"{len(df_plddt)}"
    )


    # ========================================================
    # SECONDARY STRUCTURE
    # ========================================================

    print(
        "\nRunning pydssp for secondary structure..."
    )


    with open(
        PDB_FILE,
        "r"
    ) as f:

        pdb_text = f.read()


    coords = pydssp.read_pdbtext(
        pdb_text
    )


    ss_results = pydssp.assign(

        coords,

        out_type="c3"

    )


    ss_results = list(
        ss_results
    )


    if len(ss_results) != len(df_plddt):

        raise ValueError(

            "Secondary structure length does not "
            "match protein residue count.\n"

            f"Protein residues: "
            f"{len(df_plddt)}\n"

            f"Secondary structure values: "
            f"{len(ss_results)}"

        )


    df_plddt[
        "secondary_structure"
    ] = ss_results


    total = len(
        df_plddt
    )


    helix = (

        df_plddt[
            "secondary_structure"
        ]

        == "H"

    ).sum()


    sheet = (

        df_plddt[
            "secondary_structure"
        ]

        == "E"

    ).sum()


    coil = (

        df_plddt[
            "secondary_structure"
        ]

        == "-"

    ).sum()


    print(
        f"Total residues : {total}"
    )


    if total > 0:

        print(
            f"Helix          : "
            f"{helix} "
            f"({100 * helix / total:.1f}%)"
        )

        print(
            f"Sheet          : "
            f"{sheet} "
            f"({100 * sheet / total:.1f}%)"
        )

        print(
            f"Coil           : "
            f"{coil} "
            f"({100 * coil / total:.1f}%)"
        )


    # ========================================================
    # RSA
    # ========================================================

    print(
        "\nCalculating RSA..."
    )


    parser2 = PDBParser(
        QUIET=True
    )


    structure2 = parser2.get_structure(
        "CLN2",
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

            if residue.id[0] != " ":

                continue


            if not all(

                atom in residue

                for atom in (
                    "N",
                    "CA",
                    "C",
                    "O"
                )

            ):

                continue


            residue_number = (
                residue.id[1]
            )


            resname = (
                residue.get_resname()
            )


            asa = residue.sasa


            max_val = max_asa.get(
                resname,
                200
            )


            rsa = round(
                asa / max_val,
                3
            )


            rsa_data.append({

                "residue_number":
                    residue_number,

                "RSA":
                    rsa

            })


    df_rsa = pd.DataFrame(
        rsa_data
    )


    # ========================================================
    # CONTACT COUNT
    # ========================================================

    print(
        "\nCalculating inter-residue contacts..."
    )


    parser3 = PDBParser(
        QUIET=True
    )


    structure3 = parser3.get_structure(
        "CLN2",
        PDB_FILE
    )


    atoms = list(
        structure3.get_atoms()
    )


    ns = NeighborSearch(
        atoms
    )


    contact_data = []


    for chain in structure3[0]:

        for residue in chain:

            if residue.id[0] != " ":

                continue


            if not all(

                atom in residue

                for atom in (
                    "N",
                    "CA",
                    "C",
                    "O"
                )

            ):

                continue


            residue_number = (
                residue.id[1]
            )


            ca = residue["CA"]


            neighbors = ns.search(

                ca.get_vector()
                .get_array(),

                CONTACT_RADIUS

            )


            neighbor_residues = set(

                a.get_parent().id[1]

                for a in neighbors

                if (

                    a.get_parent().id[1]

                    !=

                    residue_number

                )

            )


            contact_data.append({

                "residue_number":
                    residue_number,

                "contact_count":
                    len(
                        neighbor_residues
                    )

            })


    df_contacts = pd.DataFrame(
        contact_data
    )


    print(
        "Mean contacts per residue :",
        round(
            df_contacts[
                "contact_count"
            ].mean(),
            1
        )
    )


    print(
        "Min contacts              :",
        df_contacts[
            "contact_count"
        ].min()
    )


    print(
        "Max contacts              :",
        df_contacts[
            "contact_count"
        ].max()
    )


    # ========================================================
    # MERGE STRUCTURAL FEATURES
    # ========================================================

    print(
        "\nMerging all structural features..."
    )


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


    # ========================================================
    # BURIAL
    # ========================================================

    df_final["burial"] = (

        df_final[
            "RSA"
        ]

        .apply(

            lambda x:

            "buried"

            if x < 0.2

            else (

                "exposed"

                if x > 0.5

                else "intermediate"

            )

        )

    )


    print(
        "\nBurial distribution:"
    )


    print(
        df_final[
            "burial"
        ].value_counts()
    )


    print(
        "\nContact count summary:"
    )


    print(
        df_final[
            "contact_count"
        ].describe()
    )


    print(
        "\nSample output:"
    )


    print(

        df_final[[

            "residue_number",
            "amino_acid",
            "pLDDT",
            "secondary_structure",
            "RSA",
            "burial",
            "contact_count"

        ]]

        .head(10)

        .to_string()

    )


    # ========================================================
    # SAVE
    # ========================================================

    df_final.to_csv(

        PDB_OUTPUT_FILE,

        index=False

    )


    print(
        "\nSaved CLN2 structural features:"
    )


    print(
        PDB_OUTPUT_FILE
    )


    print(
        f"Shape: {df_final.shape}"
    )


    print(
        "Columns:",
        df_final.columns.tolist()
    )


    return df_final


# ============================================================
# LOAD STRUCTURAL FEATURES
# ============================================================

def load_structural_features():

    print(
        "\nLoading CLN2 structural features..."
    )


    run_structural_feature_extraction()


    pdb = pd.read_csv(
        PDB_OUTPUT_FILE
    )


    cols = [

        "residue_number",
        "pLDDT",
        "secondary_structure",
        "RSA",
        "burial"

    ]


    if "contact_count" in pdb.columns:

        cols.append(
            "contact_count"
        )

        print(
            "contact_count column found"
        )


    pdb = pdb[
        cols
    ]


    pdb = pdb.rename(

        columns={

            "residue_number":
                "residue_position"

        }

    )


    pdb[
        "residue_position"
    ] = pd.to_numeric(

        pdb[
            "residue_position"
        ],

        errors="coerce"

    )


    print(
        f"Structural feature entries: "
        f"{len(pdb)}"
    )


    return pdb


# ============================================================
# POSITION EXTRACTION
# ============================================================

def extract_position_from_name(name):

    name = str(name)


    # p.Glu295Lys

    match = re.search(

        r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})',

        name

    )


    if match:

        pos = int(
            match.group(2)
        )


        alt = THREE_TO_ONE.get(

            match.group(3),

            None

        )


        return pos, alt


    # p.Glu295Ter

    match = re.search(

        r'p\.([A-Z][a-z]{2})(\d+)Ter',

        name

    )


    if match:

        return int(
            match.group(2)
        ), None


    # p.Glu295fs

    match = re.search(

        r'p\.([A-Z][a-z]{2})(\d+)fs',

        name

    )


    if match:

        return int(
            match.group(2)
        ), None


    # p.T294M

    match = re.search(

        r'p\.([A-Z])(\d+)([A-Z])',

        name

    )


    if match:

        return (

            int(match.group(2)),

            match.group(3)

        )


    # p.T294*

    match = re.search(

        r'p\.([A-Z])(\d+)\*',

        name

    )


    if match:

        return int(
            match.group(2)
        ), None


    # c.123A>G

    match = re.search(

        r'c\.(\d+)[ACGT]>[ACGT]',

        name

    )


    if match:

        nt_pos = int(
            match.group(1)
        )


        aa_pos = (
            nt_pos + 2
        ) // 3


        return aa_pos, None


    # c.123del / ins / dup

    match = re.search(

        r'c\.(\d+)(?:del|ins|dup)',

        name

    )


    if match:

        nt_pos = int(
            match.group(1)
        )


        aa_pos = (
            nt_pos + 2
        ) // 3


        return aa_pos, None


    return None, None


# ============================================================
# ALPHAMISSENSE
# ============================================================

def load_alphamissense():

    print(
        "\nLoading CLN2 AlphaMissense data..."
    )


    am = pd.read_csv(

        AM_FILE,

        sep="\t"

    )


    required = [

        "position",
        "a.a.2",
        "pathogenicity score",
        "pathogenicity class"

    ]


    missing = [

        c

        for c in required

        if c not in am.columns

    ]


    if missing:

        raise ValueError(

            "Missing AlphaMissense columns: "

            + str(missing)

        )


    am = am[
        required
    ]


    am.columns = [

        "residue_position",
        "alt_aa",
        "am_pathogenicity",
        "am_class"

    ]


    am[
        "residue_position"
    ] = pd.to_numeric(

        am[
            "residue_position"
        ],

        errors="coerce"

    )


    print(
        f"AlphaMissense entries: "
        f"{len(am)}"
    )


    return am


# ============================================================
# CADD QUERY
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


        cmd = [

            "wsl",
            "bash",
            "-c",

            f"tabix "
            f"{CADD_INDEL_FILE} "
            f"{region}"

        ]


        result = subprocess.run(

            cmd,

            capture_output=True,

            text=True,

            timeout=10

        )


        for line in (

            result.stdout
            .strip()
            .split("\n")

        ):

            if not line:

                continue


            parts = line.split(
                "\t"
            )


            if len(parts) < 6:

                continue


            if (

                parts[2] == ref

                and

                parts[3] == alt

            ):

                return (

                    float(parts[4]),

                    float(parts[5])

                )


    except Exception:

        pass


    return None, None


# ============================================================
# CADD-INDEL
# ============================================================

def get_cadd_indel_scores(
    df_indels
):

    print(

        f"\nQuerying CADD-Indel "
        f"for {len(df_indels)} "
        f"indel variants..."

    )


    raw_scores = []

    phred_scores = []


    for i, (_, row) in enumerate(

        df_indels.iterrows()

    ):

        if i % 50 == 0:

            print(

                f"  Progress: "
                f"{i}/{len(df_indels)}"

            )


        chrom = str(

            row[
                "Chromosome"
            ]

        ).replace(
            "chr",
            ""
        )


        pos = int(
            row["Start"]
        )


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


        raw_scores.append(
            raw
        )


        phred_scores.append(
            phred
        )


    df_indels = df_indels.copy()


    df_indels[
        "CADD_indel_raw"
    ] = raw_scores


    df_indels[
        "CADD_indel_phred"
    ] = phred_scores


    matched = sum(

        x is not None

        for x in phred_scores

    )


    print(

        f"CADD-Indel matched: "
        f"{matched} / "
        f"{len(df_indels)}"

    )


    return df_indels


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(

    INPUT_FILE,

    OUTPUT_PREFIX

):

    print(
        "\n"
    )


    print(
        f"Processing: {INPUT_FILE}"
    )


    # ========================================================
    # LOAD ALPHAMISSENSE
    # ========================================================

    am = load_alphamissense()


    # ========================================================
    # LOAD STRUCTURE
    # ========================================================

    pdb = load_structural_features()


    # ========================================================
    # LOAD CLINVAR
    # ========================================================
    #
    # SAME LOGIC AS Preprocessingv3:
    #
    #   read_csv
    #   clean columns
    #   select required columns
    #   rename fields
    #   filter gene
    #   filter GRCh38
    #
    # No Python engine / chunking is introduced.
    #
    # ========================================================

    print(
        "\nLoading ClinVar variant summary..."
    )


    df = pd.read_csv(

        INPUT_FILE,

        sep="\t",

        low_memory=False

    )


    df.columns = (

        df.columns
        .str.strip()
        .str.replace(
            "#",
            "",
            regex=False
        )

    )


    df = df[[

        "VariationID",
        "GeneSymbol",
        "ClinicalSignificance",
        "Chromosome",
        "PositionVCF",
        "Assembly",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "Name"

    ]]


    df = df.rename(

        columns={

            "ReferenceAlleleVCF":
                "ReferenceAllele",

            "AlternateAlleleVCF":
                "AlternateAllele",

            "PositionVCF":
                "Start"

        }

    )


    # ========================================================
    # FILTER TPP1
    # ========================================================

    df = df[

        df[
            "GeneSymbol"
        ]

        .astype(str)
        .str.strip()

        == GENE

    ]


    print(
        f"CLN2 / TPP1 variants found: "
        f"{len(df)}"
    )


    # ========================================================
    # FILTER GRCh38
    # ========================================================

    df = df[

        df[
            "Assembly"
        ]

        == "GRCh38"

    ]


    print(
        f"After GRCh38 filter: "
        f"{len(df)}"
    )


    if len(df) == 0:

        print(
            "No GRCh38 CLN2 variants found."
        )

        return


    # ========================================================
    # CLINICAL SIGNIFICANCE
    # ========================================================

    df[
        "ClinicalSignificance"
    ] = (

        df[
            "ClinicalSignificance"
        ]

        .astype(str)

        .str.lower()

        .str.strip()

    )


    valid = [

        "pathogenic",
        "likely pathogenic",
        "benign",
        "likely benign"

    ]


    df = df[

        df[
            "ClinicalSignificance"
        ].isin(valid)

    ]


    # ========================================================
    # LABEL
    # ========================================================

    df["label"] = (

        df[
            "ClinicalSignificance"
        ]

        .map({

            "pathogenic": 1,

            "likely pathogenic": 1,

            "benign": 0,

            "likely benign": 0

        })

    )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    df = df.drop_duplicates(

        subset="VariationID"

    )


    print(
        f"After label filtering: "
        f"{len(df)}"
    )


    print(
        "Pathogenic:",
        (
            df["label"] == 1
        ).sum()
    )


    print(
        "Benign:",
        (
            df["label"] == 0
        ).sum()
    )


    # ========================================================
    # EXTRACT PROTEIN POSITION
    # ========================================================

    print(
        "\nExtracting variant positions..."
    )


    extracted = df[
        "Name"
    ].apply(
        extract_position_from_name
    )


    df[
        "residue_position"
    ] = extracted.apply(
        lambda x: x[0]
    )


    df[
        "alt_aa"
    ] = extracted.apply(
        lambda x: x[1]
    )


    df[
        "residue_position"
    ] = pd.to_numeric(

        df[
            "residue_position"
        ],

        errors="coerce"

    )


    print(

        "Positions extracted:",

        df[
            "residue_position"
        ].notna().sum(),

        "/",

        len(df)

    )


    # ========================================================
    # CLEAN ALLELES
    # ========================================================

    df[
        "ReferenceAllele"
    ] = (

        df[
            "ReferenceAllele"
        ]
        .astype(str)
        .str.strip()

    )


    df[
        "AlternateAllele"
    ] = (

        df[
            "AlternateAllele"
        ]
        .astype(str)
        .str.strip()

    )


    df = df[

        df[
            "ReferenceAllele"
        ].notna()

        &

        df[
            "AlternateAllele"
        ].notna()

        &

        (
            df[
                "ReferenceAllele"
            ].str.lower()
            != "na"
        )

        &

        (
            df[
                "AlternateAllele"
            ].str.lower()
            != "na"
        )

        &

        (
            df[
                "ReferenceAllele"
            ] != "."
        )

        &

        (
            df[
                "AlternateAllele"
            ] != "."
        )

        &

        (
            df[
                "ReferenceAllele"
            ] != ""
        )

        &

        (
            df[
                "AlternateAllele"
            ] != ""
        )

        &

        df[
            "Chromosome"
        ].notna()

    ]


    # ========================================================
    # SNP / INDEL
    # ========================================================

    is_snp = (

        (
            df[
                "ReferenceAllele"
            ].str.len()
            == 1
        )

        &

        (
            df[
                "AlternateAllele"
            ].str.len()
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
        f"\nSNPs   : {len(df_snps)}"
    )


    print(
        f"Indels : {len(df_indels)}"
    )


    # ========================================================
    # SNPs → ANNOVAR
    # ========================================================

    print(
        "\n--- Processing SNPs via ANNOVAR ---"
    )


    if len(df_snps) > 0:

        df_snps[
            "Chromosome"
        ] = (

            "chr"

            +

            df_snps[
                "Chromosome"
            ]
            .astype(str)
            .str.replace(
                "chr",
                "",
                regex=False
            )

        )


        annovar_input = (

            OUTPUT_PREFIX
            +
            ".avinput"

        )


        annovar_df = df_snps[[

            "Chromosome",
            "Start",
            "Start",
            "ReferenceAllele",
            "AlternateAllele"

        ]].copy()


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


        print(
            "ANNOVAR input:",
            annovar_input
        )


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

            OUTPUT_PREFIX
            .replace(
                "D:",
                "/mnt/d"
            )
            .replace(
                "\\",
                "/"
            )

        )


        cmd = [

            "wsl",
            "bash",
            "-c",

            f"cd {WSL_ANNOVAR_DIR} && "

            f"./table_annovar.pl "

            f"{wsl_input} "

            f"humandb/ "

            f"-buildver hg38 "

            f"-out {wsl_output} "

            f"-remove "

            f"-protocol "
            f"refGene,avsnp150,dbnsfp42a "

            f"-operation "
            f"g,f,f "

            f"-nastring . "

            f"-csvout "

            f"-otherinfo "

            f"-arg "
            f"'-splicing 5',,"

        ]


        print(
            "Running ANNOVAR..."
        )


        subprocess.run(
            cmd,
            check=True
        )


        ann_file = (

            OUTPUT_PREFIX
            +
            ".hg38_multianno.csv"

        )


        df_ann = pd.read_csv(
            ann_file
        )


        print(
            "Annotated SNP rows:",
            len(df_ann)
        )


        df_snps[
            "Start"
        ] = pd.to_numeric(

            df_snps[
                "Start"
            ],

            errors="coerce"

        )


        df_ann[
            "Start"
        ] = pd.to_numeric(

            df_ann[
                "Start"
            ],

            errors="coerce"

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

            how="left"

        )


        print(
            "SNPs after ANNOVAR:",
            len(df_snps)
        )


    else:

        print(
            "No SNPs to process."
        )


    # ========================================================
    # INDELS → CADD
    # ========================================================

    print(
        "\n--- Processing Indels via CADD-Indel ---"
    )


    if len(df_indels) > 0:

        df_indels[
            "Chromosome"
        ] = (

            "chr"

            +

            df_indels[
                "Chromosome"
            ]
            .astype(str)
            .str.replace(
                "chr",
                "",
                regex=False
            )

        )


        df_indels = get_cadd_indel_scores(
            df_indels
        )


    # ========================================================
    # COMBINE
    # ========================================================

    print(
        "\nCombining SNPs and Indels..."
    )


    df = pd.concat(

        [
            df_snps,
            df_indels
        ],

        ignore_index=True

    )


    print(
        "Combined dataset:",
        len(df)
    )


    # ========================================================
    # VARIANT LENGTH
    # ========================================================

    df[
        "var_length"
    ] = (

        df[
            "AlternateAllele"
        ].str.len()

        -

        df[
            "ReferenceAllele"
        ].str.len()

    )


    # ========================================================
    # ALPHAMISSENSE MERGE
    # ========================================================

    print(
        "\nMerging CLN2 AlphaMissense..."
    )


    df = df.merge(

        am,

        on=[

            "residue_position",
            "alt_aa"

        ],

        how="left"

    )


    matched_am = (

        df[
            "am_pathogenicity"
        ].notna().sum()

    )


    print(

        f"Matched to AlphaMissense: "
        f"{matched_am} / {len(df)}"

    )


    df[
        "am_likely_pathogenic"
    ] = (

        df[
            "am_class"
        ]

        ==

        "likely_pathogenic"

    ).astype(int)


    df[
        "am_ambiguous"
    ] = (

        df[
            "am_class"
        ]

        ==

        "ambiguous"

    ).astype(int)


    # ========================================================
    # STRUCTURAL MERGE
    # ========================================================

    print(
        "\nMerging CLN2 structural features..."
    )


    df = df.merge(

        pdb,

        on="residue_position",

        how="left"

    )


    matched_pdb = (

        df[
            "pLDDT"
        ].notna().sum()

    )


    print(

        f"Matched to structural features: "
        f"{matched_pdb} / {len(df)}"

    )


    # ========================================================
    # STRUCTURAL ENCODING
    # ========================================================

    df[
        "ss_helix"
    ] = (

        df[
            "secondary_structure"
        ]

        == "H"

    ).astype(int)


    df[
        "ss_sheet"
    ] = (

        df[
            "secondary_structure"
        ]

        == "E"

    ).astype(int)


    df[
        "ss_coil"
    ] = (

        df[
            "secondary_structure"
        ]

        == "-"

    ).astype(int)


    df[
        "is_buried"
    ] = (

        df[
            "burial"
        ]

        == "buried"

    ).astype(int)


    df[
        "is_exposed"
    ] = (

        df[
            "burial"
        ]

        == "exposed"

    ).astype(int)


    df[
        "is_intermediate"
    ] = (

        df[
            "burial"
        ]

        == "intermediate"

    ).astype(int)


    # ========================================================
    # FEATURES
    # ========================================================

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

        feature

        for feature in features

        if feature in df.columns

    ]


    print(
        f"\nFeatures available: "
        f"{len(features)}"
    )


    print(
        features
    )


    # ========================================================
    # KEEP LABEL
    # ========================================================

    df = df[

        features

        +

        ["label"]

    ]


    # ========================================================
    # CLEAN MISSING VALUES
    # ========================================================

    df = df.replace(
        ".",
        pd.NA
    )


    for col in features:

        df[col] = pd.to_numeric(

            df[col],

            errors="coerce"

        )


    # ========================================================
    # SPLIT SNP / INDEL AGAIN
    # ========================================================

    is_snp_row = (

        df[
            "var_length"
        ]

        == 0

    )


    is_indel_row = (

        df[
            "var_length"
        ]

        != 0

    )


    df_snps_final = df[
        is_snp_row
    ].copy()


    df_indels_final = df[
        is_indel_row
    ].copy()


    # ========================================================
    # SNP MISSING VALUE THRESHOLD
    # ========================================================

    snp_thresh = int(

        len(features)
        * 0.5

    )


    df_snps_final = (

        df_snps_final
        .dropna(
            thresh=snp_thresh
        )

    )


    # ========================================================
    # INDEL MISSING VALUE THRESHOLD
    # ========================================================

    indel_thresh = int(

        len(features)
        * 0.3

    )


    df_indels_final = (

        df_indels_final
        .dropna(
            thresh=indel_thresh
        )

    )


    # ========================================================
    # FINAL DATASET
    # ========================================================

    df = pd.concat(

        [

            df_snps_final,
            df_indels_final

        ],

        ignore_index=True

    )


    print(
        "\nFinal preprocessing results:"
    )


    print(
        f"SNPs in final dataset  : "
        f"{(df['var_length'] == 0).sum()}"
    )


    print(
        f"Indels in final dataset: "
        f"{(df['var_length'] != 0).sum()}"
    )


    print(
        "Final CLN2 dataset size:",
        df.shape
    )


    print(
        "Pathogenic:",
        (
            df["label"] == 1
        ).sum()
    )


    print(
        "Benign:",
        (
            df["label"] == 0
        ).sum()
    )


    # ========================================================
    # SAVE ML-READY DATASET
    # ========================================================

    output_file = (

        OUTPUT_PREFIX

        +

        "_CLN2_ML_READY.csv"

    )


    df.to_csv(

        output_file,

        index=False

    )


    print(
        "\nCLN2 DATASET SAVED:"
    )


    print(
        output_file
    )


    print(
        f"Shape: {df.shape}"
    )


# ============================================================
# 2025 PIPELINE
# ============================================================

run_pipeline(

    os.path.join(

        CLINVAR_DIR,

        "variant_summary_2025.txt"

    ),

    os.path.join(

        BASE_DIR,

        "CLN2_2025"

    )

)


# ============================================================
# 2026 PIPELINE
# ============================================================

run_pipeline(

    os.path.join(

        CLINVAR_DIR,

        "variant_summary_2026.txt"

    ),

    os.path.join(

        BASE_DIR,

        "CLN2_2026"

    )

)


# ============================================================
# END
# ============================================================

print(
    "\n"
)

print(
    "=" * 70
)

print(
    "CLN2 PREPROCESSING COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Output directory:"
)

print(
    BASE_DIR
)

print(
    "=" * 70
)