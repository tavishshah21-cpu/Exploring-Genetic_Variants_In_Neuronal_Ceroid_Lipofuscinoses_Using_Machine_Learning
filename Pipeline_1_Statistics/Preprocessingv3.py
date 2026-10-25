# ============================================================
# CLN3 INTEGRATED PREPROCESSING PIPELINE
# ============================================================
# Combines:
#   1. pdb_structure.py
#   2. Preprocessingv2.py
#
# Structural extraction now happens automatically inside
# preprocessing.
#
# No major preprocessing/model changes have been made.
#
# Steps:
#   AlphaFold PDB
#       ↓
#   pLDDT / Secondary Structure / RSA / Burial / Contacts
#       ↓
#   AlphaMissense
#       ↓
#   ClinVar filtering
#       ↓
#   SNP / Indel separation
#       ↓
#   ANNOVAR + dbNSFP
#       ↓
#   CADD-Indel
#       ↓
#   Merge all features
#       ↓
#   Missing-value filtering
#       ↓
#   ML-ready CSV
# ============================================================


import re
import subprocess
import pandas as pd
import pydssp

from Bio.PDB import PDBParser, ShrakeRupley, NeighborSearch


# ============================================================
# CONFIGURATION
# ============================================================

WSL_ANNOVAR_DIR = "~/annovar_work/annovar"
CADD_INDEL_FILE = "~/cadd_data/gnomad.genomes.r4.0.indel.tsv.gz"

GENE = "CLN3"


BASE_DIR = r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics"


AM_FILE = (
    r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics"
    r"\AlphaMissense-Search-Q13286.tsv"
)

PDB_FILE = (
    r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics"
    r"\AF-Q13286-F1-model_v6.pdb"
)

PDB_OUTPUT_FILE = (
    r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics"
    r"\cln3_structural_features.csv"
)


CONTACT_RADIUS = 8.0


# ============================================================
# ORIGINAL PREPROCESSING AMINO ACID CONVERSION
# ============================================================

THREE_TO_ONE = {
    'Ala': 'A',
    'Arg': 'R',
    'Asn': 'N',
    'Asp': 'D',
    'Cys': 'C',
    'Gln': 'Q',
    'Glu': 'E',
    'Gly': 'G',
    'His': 'H',
    'Ile': 'I',
    'Leu': 'L',
    'Lys': 'K',
    'Met': 'M',
    'Phe': 'F',
    'Pro': 'P',
    'Ser': 'S',
    'Thr': 'T',
    'Trp': 'W',
    'Tyr': 'Y',
    'Val': 'V'
}


# ============================================================
# ALPHAFOLD STRUCTURAL FEATURE EXTRACTION
# INTEGRATED FROM pdb_structure.py
# ============================================================

def run_structural_feature_extraction():

    print("\n")
    print("=" * 70)
    print("CLN3 STRUCTURAL FEATURE EXTRACTION")
    print("=" * 70)

    # ========================================================
    # BLOCK 1 — LOAD PDB + EXTRACT pLDDT
    # ========================================================

    print("\nLoading PDB structure...")

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure(
        "CLN3",
        PDB_FILE
    )

    model = structure[0]

    plddt_data = []

    for chain in model:

        for residue in chain:

            res_num = residue.get_id()[1]
            aa = residue.get_resname()

            if "CA" in residue:

                plddt = residue["CA"].get_bfactor()

                plddt_data.append({
                    "residue_number": res_num,
                    "amino_acid": aa,
                    "pLDDT": plddt
                })

    df_plddt = pd.DataFrame(plddt_data)

    print(
        f"Residues extracted: {df_plddt.shape[0]}"
    )


    # ========================================================
    # BLOCK 2 — SECONDARY STRUCTURE USING pydssp
    # ========================================================

    print("\nRunning pydssp for secondary structure...")

    with open(PDB_FILE) as f:
        pdb_text = f.read()

    coords = pydssp.read_pdbtext(pdb_text)

    ss_results = pydssp.assign(
        coords,
        out_type="c3"
    )

    df_plddt["secondary_structure"] = list(
        ss_results
    )

    total = len(df_plddt)

    helix = (
        df_plddt["secondary_structure"] == "H"
    ).sum()

    sheet = (
        df_plddt["secondary_structure"] == "E"
    ).sum()

    coil = (
        df_plddt["secondary_structure"] == "-"
    ).sum()

    print(f"Total residues : {total}")

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
    # BLOCK 3 — RSA
    # ========================================================

    print("\nCalculating RSA...")

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

        'ALA': 121,
        'ARG': 265,
        'ASN': 187,
        'ASP': 187,
        'CYS': 148,
        'GLN': 214,
        'GLU': 214,
        'GLY': 97,
        'HIS': 216,
        'ILE': 195,
        'LEU': 191,
        'LYS': 230,
        'MET': 203,
        'PHE': 228,
        'PRO': 154,
        'SER': 143,
        'THR': 163,
        'TRP': 264,
        'TYR': 255,
        'VAL': 165

    }


    rsa_data = []

    for chain in structure2[0]:

        for residue in chain:

            res_num = residue.get_id()[1]

            resname = residue.get_resname()

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

                "residue_number": res_num,

                "RSA": rsa

            })


    df_rsa = pd.DataFrame(
        rsa_data
    )


    # ========================================================
    # BLOCK 4 — CONTACT COUNT
    # ========================================================

    print(
        "\nCalculating inter-residue contacts..."
    )

    parser3 = PDBParser(QUIET=True)

    structure3 = parser3.get_structure(
        "CLN3",
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

            res_num = residue.get_id()[1]

            if "CA" in residue:

                ca = residue["CA"]

                neighbors = ns.search(

                    ca.get_vector().get_array(),

                    CONTACT_RADIUS

                )

                neighbor_residues = set(

                    a.get_parent().get_id()[1]

                    for a in neighbors

                    if (
                        a.get_parent().get_id()[1]
                        != res_num
                    )

                )

                contact_data.append({

                    "residue_number":
                        res_num,

                    "contact_count":
                        len(neighbor_residues)

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
    # BLOCK 5 — MERGE STRUCTURAL FEATURES
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
    # BURIAL CLASSIFICATION
    # ========================================================

    df_final["burial"] = df_final[
        "RSA"
    ].apply(

        lambda x:

        "buried"
        if x < 0.2

        else (
            "exposed"
            if x > 0.5

            else "intermediate"
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

        ]].head(10).to_string()

    )


    # ========================================================
    # SAVE STRUCTURAL FEATURES
    # ========================================================

    df_final.to_csv(

        PDB_OUTPUT_FILE,

        index=False

    )


    print(
        f"\n✅ Saved: "
        f"{PDB_OUTPUT_FILE}"
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
# ORIGINAL PREPROCESSING:
# POSITION EXTRACTION
# ============================================================

def extract_position_from_name(name):

    """
    Extract residue position and alt amino acid from
    ClinVar Name column.
    """

    name = str(name)


    # --------------------------------------------------------
    # Priority 1 — Three letter missense
    # p.Glu295Lys
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Priority 2 — Stop codon
    # --------------------------------------------------------

    match = re.search(
        r'p\.([A-Z][a-z]{2})(\d+)Ter',
        name
    )

    if match:

        pos = int(
            match.group(2)
        )

        return pos, None


    # --------------------------------------------------------
    # Priority 3 — Frameshift
    # --------------------------------------------------------

    match = re.search(
        r'p\.([A-Z][a-z]{2})(\d+)fs',
        name
    )

    if match:

        pos = int(
            match.group(2)
        )

        return pos, None


    # --------------------------------------------------------
    # Priority 4 — Single letter missense
    # p.T294M
    # --------------------------------------------------------

    match = re.search(
        r'p\.([A-Z])(\d+)([A-Z])',
        name
    )

    if match:

        pos = int(
            match.group(2)
        )

        alt = match.group(3)

        return pos, alt


    # --------------------------------------------------------
    # Priority 5 — Single letter stop
    # --------------------------------------------------------

    match = re.search(
        r'p\.([A-Z])(\d+)\*',
        name
    )

    if match:

        pos = int(
            match.group(2)
        )

        return pos, None


    # --------------------------------------------------------
    # Priority 6 — Coding position
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Priority 7 — Coding deletion/insertion
    # --------------------------------------------------------

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
# LOAD ALPHAMISSENSE
# ============================================================

def load_alphamissense():

    print(
        "\n📂 Loading AlphaMissense data..."
    )

    am = pd.read_csv(
        AM_FILE,
        sep="\t"
    )

    am = am[[
        "position",
        "a.a.2",
        "pathogenicity score",
        "pathogenicity class"
    ]]

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
        f"{am.shape[0]}"
    )

    return am


# ============================================================
# LOAD STRUCTURAL FEATURES
# ============================================================

def load_structural_features():

    print(
        "\n📂 Loading structural features..."
    )

    # IMPORTANT:
    # This is the integration point.
    # pdb_structure.py is now executed automatically.

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
            "  contact_count column found ✅"
        )

    else:

        print(
            "  contact_count not found"
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
        f"{pdb.shape[0]}"
    )


    return pdb


# ============================================================
# CADD-INDEL QUERY
# ============================================================

def query_cadd_indel(
    chrom,
    pos,
    ref,
    alt
):

    """
    Query CADD-Indel file for a single
    variant using tabix via WSL.
    """

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


            if len(parts) >= 6:

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
# GET CADD-INDEL SCORES
# ============================================================

def get_cadd_indel_scores(
    df_indels
):

    print(
        f"\n🔍 Querying CADD-Indel "
        f"for {len(df_indels)} "
        f"indel variants..."
    )


    cadd_raw_scores = []

    cadd_phred_scores = []


    for i, (_, row) in enumerate(
        df_indels.iterrows()
    ):

        if i % 50 == 0:

            print(
                f"  Progress: "
                f"{i}/{len(df_indels)}"
            )


        chrom = str(
            row["Chromosome"]
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


        cadd_raw_scores.append(
            raw
        )

        cadd_phred_scores.append(
            phred
        )


    df_indels = df_indels.copy()


    df_indels[
        "CADD_indel_raw"
    ] = cadd_raw_scores


    df_indels[
        "CADD_indel_phred"
    ] = cadd_phred_scores


    matched = sum(

        1

        for x in cadd_phred_scores

        if x is not None

    )


    print(
        f"CADD-Indel matched: "
        f"{matched} / "
        f"{len(df_indels)}"
    )


    return df_indels


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def run_pipeline(
    INPUT_FILE,
    OUTPUT_PREFIX
):

    print(
        f"\n🚀 Processing: "
        f"{INPUT_FILE}"
    )


    # ========================================================
    # LOAD ALPHAMISSENSE + STRUCTURAL FEATURES
    # ========================================================

    am = load_alphamissense()

    pdb = load_structural_features()


    # ========================================================
    # LOAD CLINVAR
    # ========================================================

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
    # FILTER CLN3
    # ========================================================

    df = df[
        df["GeneSymbol"] == GENE
    ]


    print(
        f"CLN3 variants found: "
        f"{df.shape[0]}"
    )


    # ========================================================
    # FILTER GRCh38
    # ========================================================

    df = df[
        df["Assembly"] == "GRCh38"
    ]


    print(
        f"After GRCh38 filter: "
        f"{df.shape[0]}"
    )


    if df.shape[0] == 0:

        print(
            "❌ No GRCh38 "
            "CLN3 variants found."
        )

        return


    # ========================================================
    # LABEL CREATION
    # ========================================================

    df[
        "ClinicalSignificance"
    ] = (

        df[
            "ClinicalSignificance"
        ]
        .astype(str)
        .str.lower()

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


    df["label"] = (

        df[
            "ClinicalSignificance"
        ].map({

            "pathogenic": 1,

            "likely pathogenic": 1,

            "benign": 0,

            "likely benign": 0

        })

    )


    df = df.drop_duplicates(

        subset="VariationID"

    )


    print(
        "After label filter:",
        df.shape[0]
    )

    print(
        "Pathogenic:",
        (df["label"] == 1).sum()
    )

    print(
        "Benign    :",
        (df["label"] == 0).sum()
    )


    # ========================================================
    # EXTRACT POSITION
    # ========================================================

    print(
        "\n🔗 Extracting positions "
        "from ClinVar Name column..."
    )


    df[
        "residue_position"
    ] = df["Name"].apply(

        lambda x:
        extract_position_from_name(x)[0]

    )


    df["alt_aa"] = df["Name"].apply(

        lambda x:
        extract_position_from_name(x)[1]

    )


    df[
        "residue_position"
    ] = pd.to_numeric(

        df[
            "residue_position"
        ],

        errors="coerce"

    )


    extracted = (

        df[
            "residue_position"
        ].notna().sum()

    )


    print(
        f"Positions extracted: "
        f"{extracted} / {len(df)}"
    )


    print(
        "Sample extractions:"
    )


    sample = df[
        df[
            "residue_position"
        ].notna()
    ][[
        "Name",
        "residue_position",
        "alt_aa"
    ]].head(8)


    print(
        sample.to_string()
    )


    # ========================================================
    # REMOVE INVALID VARIANTS
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

        &

        (
            df[
                "Start"
            ] > 0
        )

    ]


    df = df[

        ~df[
            "Chromosome"
        ]
        .astype(str)
        .str.contains(
            "na",
            case=False
        )

    ]


    print(
        "After removing invalid:",
        df.shape[0]
    )


    # ========================================================
    # SPLIT SNPs AND INDELS
    # ========================================================

    is_snp = (

        (
            df[
                "ReferenceAllele"
            ].str.len() == 1
        )

        &

        (
            df[
                "AlternateAllele"
            ].str.len() == 1
        )

    )


    df_snps = df[
        is_snp
    ].copy()


    df_indels = df[
        ~is_snp
    ].copy()


    print(
        f"\nSNPs   : "
        f"{len(df_snps)}"
    )

    print(
        f"Indels : "
        f"{len(df_indels)}"
    )


    # ========================================================
    # PROCESS SNPs — ANNOVAR
    # ========================================================

    print(
        "\n--- Processing SNPs via ANNOVAR ---"
    )


    df_snps[
        "Chromosome"
    ] = (

        "chr"
        +
        df_snps[
            "Chromosome"
        ].astype(str)

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
        "✅ ANNOVAR input created:",
        annovar_input
    )


    # ========================================================
    # WINDOWS PATH → WSL PATH
    # ========================================================

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


    # ========================================================
    # RUN ANNOVAR
    # ========================================================

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
        "⚙️ Running ANNOVAR..."
    )


    subprocess.run(
        cmd,
        check=True
    )


    # ========================================================
    # LOAD ANNOVAR RESULT
    # ========================================================

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
        df_ann.shape[0]
    )


    df_snps[
        "Start"
    ] = df_snps[
        "Start"
    ].astype(int)


    df_ann[
        "Start"
    ] = df_ann[
        "Start"
    ].astype(int)


    # ========================================================
    # MERGE ANNOVAR
    # ========================================================

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
        "SNPs after ANNOVAR merge:",
        df_snps.shape[0]
    )


    # ========================================================
    # PROCESS INDELS — CADD-INDEL
    # ========================================================

    print(
        "\n--- Processing Indels via CADD-Indel ---"
    )


    df_indels[
        "Chromosome"
    ] = (

        "chr"
        +
        df_indels[
            "Chromosome"
        ].astype(str)

    )


    df_indels = get_cadd_indel_scores(
        df_indels
    )


    # ========================================================
    # COMBINE SNPs + INDELS
    # ========================================================

    print(
        "\n🔗 Combining SNPs and Indels..."
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
        df.shape[0]
    )


    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    df["var_length"] = (

        df[
            "AlternateAllele"
        ].str.len()

        -

        df[
            "ReferenceAllele"
        ].str.len()

    )


    # ========================================================
    # MERGE ALPHAMISSENSE
    # ========================================================

    print(
        "\n🔗 Merging AlphaMissense..."
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


    if matched_am > 0:

        print(
            "AlphaMissense score by label:"
        )

        print(
            df.groupby(
                "label"
            )[
                "am_pathogenicity"
            ].mean()
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
    # MERGE STRUCTURAL FEATURES
    # ========================================================

    print(
        "\n🔗 Merging structural features..."
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
    # STRUCTURAL FEATURE ENCODING
    # ========================================================

    df["ss_helix"] = (

        df[
            "secondary_structure"
        ]
        ==
        "H"

    ).astype(int)


    df["ss_sheet"] = (

        df[
            "secondary_structure"
        ]
        ==
        "E"

    ).astype(int)


    df["ss_coil"] = (

        df[
            "secondary_structure"
        ]
        ==
        "-"

    ).astype(int)


    df["is_buried"] = (

        df[
            "burial"
        ]
        ==
        "buried"

    ).astype(int)


    df["is_exposed"] = (

        df[
            "burial"
        ]
        ==
        "exposed"

    ).astype(int)


    df["is_intermediate"] = (

        df[
            "burial"
        ]
        ==
        "intermediate"

    ).astype(int)


    # ========================================================
    # FEATURE SELECTION
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

        f

        for f in features

        if f in df.columns

    ]


    print(
        f"\nFeatures available: "
        f"{len(features)}"
    )


    print(
        features
    )


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


    is_snp_row = (

        df[
            "var_length"
        ] == 0

    )


    is_indel_row = (

        df[
            "var_length"
        ] != 0

    )


    df_snps_final = df[
        is_snp_row
    ].copy()


    df_indels_final = df[
        is_indel_row
    ].copy()


    # ========================================================
    # SNP MISSING-VALUE THRESHOLD
    # ========================================================

    snp_thresh = int(

        len(features)
        *
        0.5

    )


    df_snps_final = (

        df_snps_final
        .dropna(
            thresh=snp_thresh
        )

    )


    # ========================================================
    # INDEL MISSING-VALUE THRESHOLD
    # ========================================================

    indel_thresh = int(

        len(features)
        *
        0.3

    )


    df_indels_final = (

        df_indels_final
        .dropna(
            thresh=indel_thresh
        )

    )


    # ========================================================
    # FINAL COMBINATION
    # ========================================================

    df = pd.concat(

        [
            df_snps_final,
            df_indels_final
        ],

        ignore_index=True

    )


    print(
        f"\nSNPs in final dataset  : "
        f"{(df['var_length'] == 0).sum()}"
    )


    print(
        f"Indels in final dataset: "
        f"{(df['var_length'] != 0).sum()}"
    )


    print(
        "Final CLN3 dataset size:",
        df.shape
    )


    print(
        "Pathogenic:",
        (df["label"] == 1).sum()
    )


    print(
        "Benign    :",
        (df["label"] == 0).sum()
    )


    # ========================================================
    # SAVE ML-READY DATASET
    # ========================================================

    out_file = (

        OUTPUT_PREFIX
        +
        "_CLN3_ML_READY.csv"

    )


    df.to_csv(

        out_file,

        index=False

    )


    print(
        f"🎉 CLN3 DATASET SAVED: "
        f"{out_file}"
    )


# ============================================================
# RUN 2025
# ============================================================

run_pipeline(

    r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics\variant_summary_2025.txt",

    r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics\CLN3_2025"

)


# ============================================================
# RUN 2026
# ============================================================

run_pipeline(

    r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics\variant_summary_2026.txt",

    r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics\CLN3_2026"

)