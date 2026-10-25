# ==========================================
# CLN2 STRUCTURAL FEATURE EXTRACTION
# From PDB structure
#
# Extracts per-residue:
#   - pLDDT / B-factor
#   - Secondary structure (H/E/-)
#   - RSA (Relative Solvent Accessibility)
#   - Burial label (buried/intermediate/exposed)
#   - Contact count (inter-residue contacts within 8A)
#
# Reference for RSA max values:
#   Tien et al. 2013, PLoS ONE 8(11):e80635
#
# Reference for contact count methodology:
#   Westhues et al. 2023, Bioinformatics (AlphScore)
# ==========================================

import pydssp
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, ShrakeRupley, NeighborSearch

PDB_FILE    = r"AF-O14773-F1-model_v6.pdb"
OUTPUT_FILE = r"cln2_structural_features.csv"

CONTACT_RADIUS = 8.0  # Angstroms — from AlphScore paper


# ==========================================
# BLOCK 1 — LOAD PDB + EXTRACT B-FACTOR
# ==========================================
print("Loading PDB structure...")

parser    = PDBParser(QUIET=True)
structure = parser.get_structure("CLN2", PDB_FILE)
model     = structure[0]

plddt_data = []

for chain in model:
    for residue in chain:
        res_num = residue.get_id()[1]
        aa      = residue.get_resname()

        if "CA" in residue:
            plddt = residue["CA"].get_bfactor()

            plddt_data.append({
                "residue_number": res_num,
                "amino_acid":     aa,
                "pLDDT":          plddt
            })

df_plddt = pd.DataFrame(plddt_data)

print(f"Residues extracted: {df_plddt.shape[0]}")


# ==========================================
# BLOCK 2 — SECONDARY STRUCTURE via pydssp
# H = helix, E = sheet, - = coil
# ==========================================
print("\nRunning pydssp for secondary structure...")

coords     = pydssp.read_pdbtext(open(PDB_FILE).read())
ss_results = pydssp.assign(coords, out_type="c3")

df_plddt["secondary_structure"] = list(ss_results)

total = len(df_plddt)
helix = (df_plddt["secondary_structure"] == "H").sum()
sheet = (df_plddt["secondary_structure"] == "E").sum()
coil  = (df_plddt["secondary_structure"] == "-").sum()

print(f"Total residues : {total}")
print(f"Helix          : {helix} ({100*helix/total:.1f}%)")
print(f"Sheet          : {sheet} ({100*sheet/total:.1f}%)")
print(f"Coil           : {coil} ({100*coil/total:.1f}%)")


# ==========================================
# BLOCK 3 — RSA via BioPython ShrakeRupley
# Max ASA values from Tien et al. 2013
# PLoS ONE 8(11):e80635
#
# burial:
#   buried       < 0.2
#   intermediate 0.2–0.5
#   exposed      > 0.5
# ==========================================
print("\nCalculating RSA...")

parser2    = PDBParser(QUIET=True)
structure2 = parser2.get_structure("CLN2", PDB_FILE)

sr = ShrakeRupley()
sr.compute(structure2, level="R")

max_asa = {
    'ALA': 121, 'ARG': 265, 'ASN': 187, 'ASP': 187, 'CYS': 148,
    'GLN': 214, 'GLU': 214, 'GLY': 97,  'HIS': 216, 'ILE': 195,
    'LEU': 191, 'LYS': 230, 'MET': 203, 'PHE': 228, 'PRO': 154,
    'SER': 143, 'THR': 163, 'TRP': 264, 'TYR': 255, 'VAL': 165
}

rsa_data = []

for chain in structure2[0]:
    for residue in chain:

        res_num = residue.get_id()[1]
        resname = residue.get_resname()
        asa     = residue.sasa

        max_val = max_asa.get(resname, 200)
        rsa     = round(asa / max_val, 3)

        rsa_data.append({
            "residue_number": res_num,
            "RSA":            rsa
        })

df_rsa = pd.DataFrame(rsa_data)


# ==========================================
# BLOCK 4 — CONTACT COUNT
# Count unique residues within 8A of each CA atom
# From AlphScore paper (Westhues et al. 2023)
# ==========================================
print("\nCalculating inter-residue contacts...")

parser3    = PDBParser(QUIET=True)
structure3 = parser3.get_structure("CLN2", PDB_FILE)

atoms = list(structure3.get_atoms())
ns    = NeighborSearch(atoms)

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
                if a.get_parent().get_id()[1] != res_num
            )

            contact_data.append({
                "residue_number": res_num,
                "contact_count":  len(neighbor_residues)
            })

df_contacts = pd.DataFrame(contact_data)

print(
    f"Mean contacts per residue : "
    f"{round(df_contacts['contact_count'].mean(), 1)}"
)

print(
    f"Min contacts              : "
    f"{df_contacts['contact_count'].min()}"
)

print(
    f"Max contacts              : "
    f"{df_contacts['contact_count'].max()}"
)


# ==========================================
# BLOCK 5 — MERGE EVERYTHING
# One row per amino acid — full structural
# fingerprint of CLN2
# ==========================================
print("\nMerging all structural features...")

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
        if x < 0.2
        else (
            "exposed"
            if x > 0.5
            else "intermediate"
        )
)


# ==========================================
# SUMMARY
# ==========================================
print("\nBurial distribution:")
print(df_final["burial"].value_counts())

print("\nContact count summary:")
print(df_final["contact_count"].describe())

print("\nSample output:")

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


# ==========================================
# SAVE
# ==========================================
df_final.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved: {OUTPUT_FILE}")
print(f"Shape: {df_final.shape}")
print(f"Columns: {df_final.columns.tolist()}")