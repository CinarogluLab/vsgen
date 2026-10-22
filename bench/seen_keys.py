"""Training-set exclusion list for the held-out pKa scores (pka_eval.py --exclude-keys).

Writes the InChIKey connectivity blocks (FragmentParent + Uncharger, as
pka_eval._conn_key) of every molecule in the published training sets of
pKaSolver, Uni-pKa and QupKake. MolGpKa's and Dimorphite-DL's training data are
not included, so "held out" means held out from those three tools only.

Sources (clone each repository):
  pKaSolver  data/Baltruschat/combined_training_datasets_unique.sdf
  Uni-pKa    dataset/dwar-iBond.tsv  (SMILES column, reactions as A,B>>C,D)
  QupKake    data/exp_training_data.sdf

usage: python bench/seen_keys.py PKASOLVER_SDF UNIPKA_TSV QUPKAKE_SDF OUT [--bench SMILES_FILE]
"""

import argparse
import csv
import re

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
_UN = rdMolStandardize.Uncharger()


def key(m):
    try:
        return Chem.MolToInchiKey(_UN.uncharge(rdMolStandardize.FragmentParent(m))).split("-")[0]
    except Exception:
        return None


def from_sdf(path):
    return {key(m) for m in Chem.SDMolSupplier(path) if m is not None} - {None}


def from_tsv(path):
    out = set()
    for row in csv.DictReader(open(path), delimiter="\t"):
        for part in re.split(r">>|,", row["SMILES"]):
            m = Chem.MolFromSmiles(part)
            if m is not None:
                out.add(key(m))
    return out - {None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkasolver_sdf")
    ap.add_argument("unipka_tsv")
    ap.add_argument("qupkake_sdf")
    ap.add_argument("out")
    ap.add_argument("--bench", help="benchmark SMILES (pka_eval.py export), to report overlap per tool")
    a = ap.parse_args()
    sets = {"pkasolver": from_sdf(a.pkasolver_sdf), "unipka": from_tsv(a.unipka_tsv), "qupkake": from_sdf(a.qupkake_sdf)}
    seen = set().union(*sets.values())
    open(a.out, "w").write("\n".join(sorted(seen)) + "\n")
    print(f"{len(seen)} keys -> {a.out}")
    if a.bench:
        bench = [s.strip() for s in open(a.bench) if s.strip()]
        bk = {s: key(Chem.MolFromSmiles(s)) for s in bench}
        for n, ks in sets.items():
            print(f"{n:10s} {len(ks):6d} training molecules; benchmark molecules seen {sum(bk[s] in ks for s in bench)}/{len(bench)}")
        print(f"benchmark molecules in none of the three: {sum(bk[s] not in seen for s in bench)}")


if __name__ == "__main__":
    main()
