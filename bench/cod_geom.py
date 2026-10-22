"""3D geometry accuracy against small-molecule crystal structures (COD).

fetch: sample drug-like, single-component, neutral organic COD entries
       (allcod.smi), download their CIFs, keep well-refined ordered structures
       (R1 <= 0.05, no partial occupancy or disorder groups) whose asymmetric
       unit holds one whole molecule, and assign bond orders from the COD
       SMILES onto the crystal heavy atoms. Writes REF.sdf.
eval:  build vsgen models (3D stage only, from the crystal's own SMILES and
       stereo) and compare heavy-atom geometry with the crystal: bond lengths,
       angles, amide/ester planarity, ring-system conformation, and whole-
       molecule RMSD. Hydrogens are ignored (X-ray H positions are imprecise).

The crystal is a solid-state conformation: local geometry (bonds, angles,
planarity, ring puckers) is a fair reference; the global shape is affected by
packing, so whole-molecule RMSD is reported but not used as a pass criterion.

usage: python bench/cod_geom.py fetch allcod.smi REF.sdf [--n 400] [--cif-dir DIR]
       python bench/cod_geom.py eval REF.sdf [--force-fields uff,mmff94s]
"""

import argparse
import os
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDetermineBonds, rdMolAlign, rdMolTransforms

RDLogger.DisableLog("rdApp.*")
ELEMENTS = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}


def candidates(smi_path, n_heavy=(10, 40)):
    out = []
    for line in open(smi_path):
        parts = line.split()
        if len(parts) < 2 or "." in parts[0]:
            continue
        m = Chem.MolFromSmiles(parts[0])
        if m is None or not n_heavy[0] <= m.GetNumHeavyAtoms() <= n_heavy[1]:
            continue
        if {a.GetSymbol() for a in m.GetAtoms()} - ELEMENTS or "C" not in {a.GetSymbol() for a in m.GetAtoms()}:
            continue
        if Chem.GetFormalCharge(m) != 0:
            continue  # net-neutral only (nitro groups and zwitterions as written by COD are kept)
        out.append((parts[1], parts[0]))
    return out


def _skeleton(mol):
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    return m


def cif_to_mol(cif_text, smiles):
    """Heavy-atom RDKit molecule with crystal coordinates and SMILES bond orders, or (None, reason)."""
    import gemmi

    doc = gemmi.cif.read_string(cif_text)
    block = doc.sole_block()
    r = block.find_value("_refine_ls_R_factor_gt")
    try:
        r1 = float(r.split("(")[0]) if r not in (None, "?", ".") else None
    except ValueError:
        r1 = None
    if r1 is None or r1 > 0.05:
        return None, "R1 missing or > 0.05"
    if block.find_values("_atom_site_disorder_group") and any(
            v not in (".", "?") for v in block.find_values("_atom_site_disorder_group")):
        return None, "disorder"
    st = gemmi.make_small_structure_from_block(block)
    if any(s.occ < 0.999 for s in st.sites):
        return None, "partial occupancy"
    heavy = [s for s in st.sites if s.element.name != "H"]
    template = Chem.MolFromSmiles(smiles)
    if len(heavy) != template.GetNumAtoms():
        return None, "asymmetric unit is not one whole molecule"
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(heavy))
    for i, s in enumerate(heavy):
        rw.AddAtom(Chem.Atom(s.element.name))
        p = st.cell.orthogonalize(s.fract)
        conf.SetAtomPosition(i, (p.x, p.y, p.z))
    conn = rw.GetMol()
    conn.AddConformer(conf, assignId=True)
    try:
        rdDetermineBonds.DetermineConnectivity(conn, useHueckel=False)
    except Exception:
        return None, "connectivity perception failed"
    # Match the SMILES atom graph onto the crystal connectivity (bond orders,
    # charges and H ignored), then put the crystal coordinates on the SMILES
    # molecule itself, so chemistry comes from the SMILES and geometry from the crystal.
    match = _skeleton(conn).GetSubstructMatch(_skeleton(template))
    if len(match) != template.GetNumAtoms() or conn.GetNumBonds() != template.GetNumBonds():
        return None, "connectivity does not match the SMILES"
    mol = Chem.Mol(template)
    c = Chem.Conformer(mol.GetNumAtoms())
    pos = conn.GetConformer().GetPositions()
    for t, i in enumerate(match):
        c.SetAtomPosition(t, pos[i].tolist())
    c.Set3D(True)
    mol.AddConformer(c, assignId=True)
    Chem.AssignStereochemistryFrom3D(mol)
    temp = block.find_value("_diffrn_ambient_temperature") or block.find_value("_cell_measurement_temperature")
    mol.SetProp("R1", str(r1))
    mol.SetProp("temperature_K", (temp or "?").split("(")[0])
    return mol, None


def fetch(a):
    cands = candidates(a.smi)
    random.Random(0).shuffle(cands)
    os.makedirs(a.cif_dir, exist_ok=True)

    def get(item):
        cid, smi = item
        path = os.path.join(a.cif_dir, f"{cid}.cif")
        if not os.path.exists(path):
            try:
                text = urllib.request.urlopen(f"https://www.crystallography.net/cod/{cid}.cif", timeout=60).read()
                open(path, "wb").write(text)
            except Exception:
                return cid, smi, None, "download failed"
        try:
            mol, why = cif_to_mol(open(path, errors="replace").read(), smi)
        except Exception as exc:
            mol, why = None, f"parse error {type(exc).__name__}"
        return cid, smi, mol, why

    kept, reasons, w = 0, {}, Chem.SDWriter(a.ref)
    with ThreadPoolExecutor(4) as ex:  # polite to the COD server
        for start in range(0, len(cands), 200):
            for cid, smi, mol, why in ex.map(get, cands[start:start + 200]):
                if mol is None:
                    reasons[why] = reasons.get(why, 0) + 1
                    continue
                mol.SetProp("_Name", f"COD{cid}")
                mol.SetProp("COD_SMILES", smi)
                w.write(mol)
                kept += 1
            print(f"  tried {start + 200}, kept {kept}", flush=True)
            if kept >= a.n:
                break
    w.close()
    print(f"kept {kept}; rejected: {reasons}")


# ---------- evaluation ----------

AMIDE_ESTER = Chem.MolFromSmarts("[#6,#1:1][CX3:2](=[OX1])-[NX3,OX2:3]-[#6:4]")


def _non_aromatic_ring_systems(m):
    rings = [set(r) for r in m.GetRingInfo().AtomRings() if not all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    systems = []
    for r in rings:
        for s in [s for s in systems if len(s & r) >= 2]:
            systems.remove(s)
            r |= s
        systems.append(r)
    return [sorted(s) for s in systems]


def _kabsch(p, q):
    p, q = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    rot = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    return float(np.sqrt(((p @ rot - q) ** 2).sum(1).mean()))


def _measure(job):
    block, ff = job
    from vsgen.chem import variant_seed
    from vsgen.conf import Conf3D, build_cpu

    crys = Chem.MolFromMolBlock(block)
    smi = Chem.MolToSmiles(crys)
    models, err = build_cpu(smi, variant_seed(42, smi), Conf3D(force_field=ff))
    if not models:
        return None
    cpos = crys.GetConformer().GetPositions()
    systems = _non_aromatic_ring_systems(crys)
    out = {"name": crys.GetProp("_Name"), "bond": [], "angle": [], "planarity": [], "ring": [], "rmsd": []}
    for k, (model, _, _) in enumerate(models):
        heavy = Chem.RemoveHs(model)
        mt = heavy.GetSubstructMatch(crys)  # crystal atom i -> model atom mt[i]
        if not mt:
            return None
        mpos = heavy.GetConformer().GetPositions()[list(mt)]
        if k == 0:  # local geometry of the kept (lowest-energy) model
            for b in crys.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                out["bond"].append((str(b.GetBondType()), np.linalg.norm(mpos[i] - mpos[j]) - np.linalg.norm(cpos[i] - cpos[j])))
            for a in crys.GetAtoms():
                nb = [n.GetIdx() for n in a.GetNeighbors()]
                for x in range(len(nb)):
                    for y in range(x + 1, len(nb)):
                        def ang(p):
                            u, v = p[nb[x]] - p[a.GetIdx()], p[nb[y]] - p[a.GetIdx()]
                            return np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
                        out["angle"].append(ang(mpos) - ang(cpos))
            for q in crys.GetSubstructMatches(AMIDE_ESTER):
                def tors(p):
                    b1, b2, b3 = p[q[1]] - p[q[0]], p[q[2]] - p[q[1]], p[q[3]] - p[q[2]]
                    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
                    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
                    return np.degrees(np.arctan2(m1 @ n2, n1 @ n2))
                tm, tc = tors(mpos), tors(cpos)
                out["planarity"].append((min(abs(tm), 180 - abs(tm)), (abs(tm) > 90) == (abs(tc) > 90)))
        out["ring"].append(max((_kabsch(mpos[s], cpos[s]) for s in systems), default=None) if systems else None)
        ref = Chem.Mol(crys)
        out["rmsd"].append(rdMolAlign.GetBestRMS(Chem.Mol(heavy), ref))
    return out


def evaluate(a):
    blocks = [Chem.MolToMolBlock(m) for m in Chem.SDMolSupplier(a.ref) if m is not None]
    print(f"{len(blocks)} COD structures")
    for ff in a.force_fields.split(","):
        with Pool(16) as p:
            res = [r for r in p.map(_measure, [(b, ff) for b in blocks], chunksize=4) if r]
        bonds = [d for r in res for _, d in r["bond"]]
        by_type = {}
        for r in res:
            for t, d in r["bond"]:
                by_type.setdefault(t, []).append(abs(d))
        angles = np.abs([d for r in res for d in r["angle"]])
        plan = [p for r in res for p in r["planarity"]]
        ring_best = [min(x for x in r["ring"] if x is not None) for r in res if any(x is not None for x in r["ring"])]
        ring_first = [r["ring"][0] for r in res if r["ring"][0] is not None]
        rmsd_first = [r["rmsd"][0] for r in res]
        rmsd_best = [min(r["rmsd"]) for r in res]
        print(f"\n== {ff}: {len(res)} molecules")
        print(f"bond length |error|: mean {np.mean(np.abs(bonds)):.4f} A, signed mean {np.mean(bonds):+.4f} A, "
              f"95th pct {np.percentile(np.abs(bonds), 95):.3f} A  (n={len(bonds)})")
        print("   by type: " + ", ".join(f"{t.lower()} {np.mean(v):.4f} (n={len(v)})" for t, v in sorted(by_type.items()) if len(v) >= 20))
        print(f"bond angle |error|: mean {angles.mean():.2f} deg, 95th pct {np.percentile(angles, 95):.1f} deg")
        if plan:
            print(f"amide/ester C-X torsion: mean deviation from planar {np.mean([p[0] for p in plan]):.1f} deg, "
                  f"> 15 deg: {np.mean([p[0] > 15 for p in plan]):.1%}, cis/trans as in crystal: {np.mean([p[1] for p in plan]):.1%} (n={len(plan)})")
        print(f"ring systems (worst per molecule): kept model <= 0.3 A {np.mean(np.array(ring_first) <= 0.3):.1%}, "
              f"best of models <= 0.3 A {np.mean(np.array(ring_best) <= 0.3):.1%}, median kept {np.median(ring_first):.3f} A (n={len(ring_first)})")
        print(f"whole-molecule heavy-atom RMSD (packing-affected): kept model median {np.median(rmsd_first):.2f} A, "
              f"best of models median {np.median(rmsd_best):.2f} A, <= 1 A: {np.mean(np.array(rmsd_best) <= 1):.1%}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("smi")
    f.add_argument("ref")
    f.add_argument("--n", type=int, default=400)
    f.add_argument("--cif-dir", default="cod_cif")
    e = sub.add_parser("eval")
    e.add_argument("ref")
    e.add_argument("--force-fields", default="uff,mmff94s")
    a = ap.parse_args()
    fetch(a) if a.cmd == "fetch" else evaluate(a)


if __name__ == "__main__":
    main()
