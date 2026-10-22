"""External pKa test: Novartis (in-house measurements) and SAMPL6/7/8 (blind
challenges), as shipped in the Uni-pKa repository's dataset/ folder. Not part
of IUPAC or DataWarrior, so it checks for leakage in IUPAC-based numbers.
Caveats: the official Uni-pKa checkpoint used Novartis as its validation set;
pkasolver and QupKake used Novartis only as a test set.

Each entry is a measured macro pKa for a transition between macrostates of
charge qA -> qB = qA - 1. Scored transitions: 0 -> -1 (acids) and +1 -> 0
(bases), predicted as
  sites tools:  acid -log10(sum 10^-pKa_i) over acid sites, base log10(sum 10^pKa_j)
  states tools: the tool's own macro pKa for that transition.

usage: python bench/pka_external.py export UNIPKA_DATASET_DIR smiles.txt
       python bench/pka_external.py score UNIPKA_DATASET_DIR NAME=pred.jsonl [...]   (NAME=molgpka is computed here)
"""

import csv
import hashlib
import json
import math
import os
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
SETS = ["novartis_acid", "novartis_base", "sampl6", "sampl7", "sampl8"]


def entries(dataset_dir):
    """(set, neutral SMILES, transition 'acid'|'base'|None, experimental pKa)."""
    out = []
    for name in SETS:
        for row in csv.DictReader(open(os.path.join(dataset_dir, f"{name}.tsv")), delimiter="\t"):
            a, b = row["SMILES"].split(">>")
            qa = Chem.GetFormalCharge(Chem.MolFromSmiles(a.split(",")[0]))
            qb = Chem.GetFormalCharge(Chem.MolFromSmiles(b.split(",")[0]))
            kind = {(0, -1): "acid", (1, 0): "base"}.get((qa, qb))
            micro = [s for s in (a + "," + b).split(",") if Chem.GetFormalCharge(Chem.MolFromSmiles(s)) == 0]
            base = micro[0] if micro else a.split(",")[0]
            neutral = Chem.MolToSmiles(rdMolStandardize.Uncharger().uncharge(Chem.MolFromSmiles(base)))
            out.append((name, neutral, kind, float(row["TARGET"])))
    return out


def macro(rec, kind):
    if not rec or rec.get("error"):
        return None
    if rec["kind"] == "states" and rec.get("macro"):
        return rec["macro"].get(kind)
    sites = list((rec.get("acids", {}) if kind == "acid" else rec.get("bases", {})).values())
    if not sites:
        return None
    if kind == "acid":
        return -math.log10(sum(10 ** -p for p in sites))
    return math.log10(sum(10 ** p for p in sites))


def _dimorphite_table():
    import dimorphite_dl

    return os.path.join(os.path.dirname(dimorphite_dl.__file__), "smarts", "site_substructures.smarts")


def dimorphite_version():
    """Version and SMARTS-table hash: Dimorphite's behavior changes between releases
    (2.1.0 added Phenol_unactivated; Gypsum-DL 1.3.0 is often installed with 2.0.x)."""
    from importlib.metadata import version

    digest = hashlib.sha256(open(_dimorphite_table(), "rb").read()).hexdigest()[:12]
    return f"dimorphite_dl {version('dimorphite_dl')} (table {digest})"


def dimorphite_records(smiles):
    """Per-site pKa from Dimorphite-DL's own table, so it can be scored like
    the ML predictors. Rows are name/SMARTS/(index, mean, stdev)...; direction
    follows its charge rule (protonate/change.py set_protonation_charge): a
    nitrogen site carries the conjugate-acid pKa of a base, unless the entry
    name starts with "*" (its N-acids: amides, sulfonamides, imides); every
    other site is an acid. Matched atoms are claimed by the first pattern, as
    Dimorphite protects them."""
    table = []
    for line in open(_dimorphite_table()):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        query = Chem.MolFromSmarts(parts[1])
        if query is None:
            continue
        pkas = [(int(parts[i]), float(parts[i + 1])) for i in range(2, len(parts) - 1, 3)]
        table.append((parts[0], query, pkas))
    recs = {}
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        mol_h = Chem.AddHs(mol)  # its patterns contain explicit [H], as Dimorphite matches with H added
        acids, bases, claimed = {}, {}, set()
        for name, query, pkas in table:
            for match in mol_h.GetSubstructMatches(query):
                for idx, mean in pkas:
                    if idx >= len(match) or mean < -100:
                        continue
                    site = match[idx]
                    atom = mol_h.GetAtomWithIdx(site)
                    if atom.GetAtomicNum() == 1:  # a pattern can point at the acidic H itself
                        site = atom.GetNeighbors()[0].GetIdx()
                        atom = mol_h.GetAtomWithIdx(site)
                    if site in claimed or site >= mol.GetNumAtoms():
                        continue
                    claimed.add(site)
                    if atom.GetSymbol() == "N" and not name.startswith("*"):
                        bases[site] = mean
                    else:
                        acids[site] = mean
        recs[smi] = {"kind": "sites", "acids": acids, "bases": bases}
    return recs


def molgpka_records(smiles):
    from vsgen import pka

    recs = {}
    for s in smiles:
        _, a, b = pka.predict(Chem.MolFromSmiles(s))
        recs[s] = {"kind": "sites", "acids": a, "bases": b}
    return recs


def main():
    cmd, dataset_dir = sys.argv[1], sys.argv[2]
    data = entries(dataset_dir)
    if cmd == "export":
        with open(sys.argv[3], "w") as f:
            f.write("\n".join(dict.fromkeys(s for _, s, _, _ in data)) + "\n")
        print(f"{len(data)} entries, {len({s for _, s, _, _ in data})} molecules; "
              f"scored transitions: {sum(k is not None for *_, k, _ in data)}")
        return
    print(f"{'tool':18s} " + " ".join(f"{n:>22s}" for n in ["novartis (n)", "sampl6-8 (n)", "all: MAE RMSE <=1"]))
    per_tool = {}
    for arg in sys.argv[3:]:
        name, _, path = arg.partition("=")
        if name == "molgpka":
            recs = molgpka_records({s for _, s, _, _ in data})
        elif name == "dimorphite":
            recs = dimorphite_records({s for _, s, _, _ in data})
            print(f"  ({dimorphite_version()})")
        else:
            recs = {}
            for line in open(path):
                r = json.loads(line)
                recs[r["smiles"]] = r
        groups = {"novartis": [], "sampl": [], "all": []}
        missing = 0
        per_tool[name] = {}
        for k, (set_name, smi, kind, target) in enumerate(data):
            if kind is None:
                continue
            p = macro(recs.get(smi), kind)
            if p is None:
                missing += 1
                continue
            err = abs(p - target)
            per_tool[name][k] = err
            groups["novartis" if set_name.startswith("novartis") else "sampl"].append(err)
            groups["all"].append(err)
        cell = lambda e: f"{np.mean(e):.2f} ({len(e)})" if e else "-"
        e = np.array(groups["all"])
        print(f"{name:18s} {cell(groups['novartis']):>22s} {cell(groups['sampl']):>22s} "
              f"{e.mean():6.2f} {np.sqrt((e ** 2).mean()):5.2f} {np.mean(e <= 1):6.1%}   no prediction: {missing}")
    common = set.intersection(*(set(v) for v in per_tool.values()))
    print(f"\nentries every tool predicts: {len(common)}")
    for name, errs in per_tool.items():
        e = np.array([errs[k] for k in sorted(common)])
        print(f"{name:18s} MAE {e.mean():.2f}  RMSE {np.sqrt((e ** 2).mean()):.2f}  <=1 {np.mean(e <= 1):.1%}")


if __name__ == "__main__":
    main()
