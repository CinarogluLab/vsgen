"""Dock all models of one complex with Vina-ACO (run in an env with Meeko).

Receptor via Open Babel (pH 7.4 polar H; Meeko's receptor prep rejects the
metals, cofactors and glycans common in Astex pockets), box = crystal ligand
extent + 8 A padding per side, ligands via Meeko, one `vina_aco --batch` per seed (grids built once), top
pose of each model exported back to SDF. Called by dockval.py.

usage: python aco_dock.py PROTEIN.pdb CRYSTAL_LIGAND.sdf OUTDIR SEEDS MODEL.sdf [...]
       SEEDS: comma-separated, e.g. 1,2,3. Writes OUTDIR/results.json.
"""

import json
import os
import subprocess
import sys

from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate
from rdkit import Chem

VINA_ACO = os.environ.get("VINA_ACO", "vina_aco")
OBABEL = os.environ.get("OBABEL", "obabel")
PRESET = os.environ.get("ACO_PRESET", "accurate")


def main():
    protein, crystal, out, seeds, models = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","), sys.argv[5:]
    os.makedirs(out, exist_ok=True)
    rec = os.path.join(out, "rec")
    if not os.path.exists(rec + ".pdbqt"):
        subprocess.run([OBABEL, "-ipdb", protein, "-p", "7.4", "-xr", "-opdbqt", "-O", rec + ".pdbqt"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pos = Chem.MolFromMolFile(crystal).GetConformer().GetPositions()
        lo, hi = pos.min(0), pos.max(0)
        with open(rec + ".box.txt", "w") as f:
            for ax, c, size in zip("xyz", (lo + hi) / 2, hi - lo + 16):
                f.write(f"center_{ax} = {c:.3f}\nsize_{ax} = {size:.3f}\n")
    res, ligs = {}, []
    for path in models:
        name = os.path.splitext(os.path.basename(path))[0]
        mol = Chem.MolFromMolFile(path, removeHs=False)
        try:
            text, ok, err = PDBQTWriterLegacy.write_string(MoleculePreparation().prepare(mol)[0])
        except Exception as exc:
            ok, err = False, str(exc)
        res[name] = {"meeko_ok": bool(ok), "meeko_error": None if ok else str(err)[:120], "seeds": {}}
        if ok:
            p = os.path.join(out, name + ".pdbqt")
            open(p, "w").write(text)
            ligs.append(p)
    for seed in seeds:
        d = os.path.join(out, f"seed{seed}")
        os.makedirs(d, exist_ok=True)
        if ligs:
            subprocess.run([VINA_ACO, "--receptor", rec + ".pdbqt", "--batch", *ligs, "--config", rec + ".box.txt",
                            "--search", "aco", "--aco_preset", PRESET, "--cpu", "1", "--seed", seed, "--dir", d],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
        for p in ligs:
            name = os.path.splitext(os.path.basename(p))[0]
            o = os.path.join(d, name + "_out.pdbqt")
            if not os.path.exists(o) or os.path.getsize(o) == 0:
                continue
            score = float(next(l for l in open(o) if l.startswith("REMARK VINA RESULT")).split()[3])
            pose = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule.from_file(o, skip_typing=True))[0]
            top = Chem.Mol(pose, confId=pose.GetConformers()[0].GetId())
            sdf = os.path.join(d, name + "_top.sdf")
            Chem.MolToMolFile(top, sdf)
            res[name]["seeds"][seed] = {"score": score, "sdf": sdf}
    json.dump(res, open(os.path.join(out, "results.json"), "w"))


if __name__ == "__main__":
    main()
