"""Docking-readiness validation against crystal structures.

For each protein-ligand complex (dirs holding <id>_ligand.sdf and
<id>_protein.pdb, e.g. the Astex Diverse and PoseBusters Benchmark sets), the
crystal ligand becomes a neutral, stereo-defined SMILES; each preparation
method turns it into 3D models, which are then checked three ways:

  validity  PoseBusters ligand checks (bond lengths/angles, planarity, clashes,
            internal energy) on every model
  rings     non-aromatic ring-system RMSD of the best model vs the crystal;
            docking keeps rings rigid, so a wrong ring pucker cannot be fixed
  redock    smina into the crystal pocket; success = top-scored pose across
            all of a method's models within 2 A heavy-atom RMSD of the crystal

Methods: vsgen (default settings), gypsum (Gypsum-DL defaults), rdkit
(naive control: one ETKDG+MMFF conformer of the neutral input, no protonation)
and crystal (the crystal ligand + H: the ceiling of this docking protocol).
Stages are cached in WORKDIR, so an interrupted run resumes.

usage: python bench/dockval.py WORKDIR DATASET_DIR [...] [--methods crystal,vsgen,gypsum,rdkit] [--ids FILE] [--limit N]
"""

import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import time
from multiprocessing import get_context

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDistGeom
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
SMINA = os.environ.get("SMINA", "smina")
GYPSUM = os.environ.get("GYPSUM", "gypsum-dl")  # release 1.3.0 (no --random_seed)
ACO_PY = os.environ.get("ACO_PYTHON", "python")  # an environment with Meeko
MOLSCRUB_PY = os.environ.get("MOLSCRUB_PYTHON", "python")  # an environment with molscrub


# ---------- inputs ----------

def build_inputs(work, dataset_dirs, limit, ids=None, strip_stereo=False):
    man_path = os.path.join(work, "manifest.json")
    if os.path.exists(man_path):
        return json.load(open(man_path))
    manifest = {}
    for d in dataset_dirs:
        for lig in sorted(glob.glob(os.path.join(d, "**", "*_ligand.sdf"), recursive=True)):
            cid = os.path.basename(lig)[: -len("_ligand.sdf")]
            prot = os.path.join(os.path.dirname(lig), f"{cid}_protein.pdb")
            m = Chem.MolFromMolFile(lig)
            if m is None or not os.path.exists(prot) or cid in manifest or (ids and cid not in ids):
                continue
            Chem.AssignStereochemistryFrom3D(m)
            neutral = rdMolStandardize.Uncharger().uncharge(m)
            if strip_stereo:
                Chem.RemoveStereochemistry(neutral)
            smi = Chem.MolToSmiles(neutral)
            manifest[cid] = {"ligand": lig, "protein": prot, "smiles": smi}
    if limit:
        manifest = dict(list(manifest.items())[:limit])
    with open(os.path.join(work, "inputs.smi"), "w") as f:
        for cid, v in manifest.items():
            f.write(f"{v['smiles']} {cid}\n")
    json.dump(manifest, open(man_path, "w"), indent=1)
    return manifest


# ---------- preparation methods ----------

def vsgen_config(gpu):
    """What a cached vsgen.sdf was made with: run settings plus a hash of the vsgen
    sources, so a code change (e.g. the force field) invalidates it."""
    import vsgen
    from vsgen.run import RunConfig

    root = os.path.dirname(vsgen.__file__)
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        h.update(open(f, "rb").read())
    return f"{RunConfig(gpu=gpu)!r} sources {h.hexdigest()[:12]}"


def prep_vsgen(work, out, gpu):
    from vsgen.run import RunConfig, run

    run(os.path.join(work, "inputs.smi"), out, RunConfig(gpu=gpu))


def prep_gypsum(work, out):
    gdir = os.path.join(work, "gypsum_out")
    os.makedirs(gdir, exist_ok=True)
    with open(os.path.join(work, "gypsum.log"), "w") as log:
        subprocess.run([GYPSUM, "--source", os.path.join(work, "inputs.smi"), "--output_folder", gdir,
                        "--num_processors", "16", "--job_manager", "multiprocessing"],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    w = Chem.SDWriter(out)
    for m in Chem.SDMolSupplier(os.path.join(gdir, "gypsum_dl_success.sdf"), removeHs=False):
        if m is not None and m.HasProp("SMILES"):  # first record holds run parameters
            w.write(m)
    w.close()


def prep_molscrub(work, out, procs=16):
    """molscrub (github.com/forlilab/molscrub), run from MOLSCRUB_PYTHON's env."""
    scrub = os.path.join(os.path.dirname(MOLSCRUB_PY), "scrub.py")
    sdf = os.path.join(work, "molscrub_raw.sdf")
    subprocess.run([MOLSCRUB_PY, scrub, os.path.join(work, "inputs.smi"), "-o", sdf,
                    "--cpu", str(procs), "--ph", "7.4"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=7200)
    w = Chem.SDWriter(out)
    for mol in Chem.SDMolSupplier(sdf, removeHs=False):
        if mol is None:
            continue
        mol.SetProp("_Name", mol.GetProp("_Name").split("_i")[0])  # "<name>_i0" -> input name
        w.write(mol)
    w.close()


def _rdkit_one(item):
    cid, smi = item
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    p = rdDistGeom.ETKDGv3()
    p.randomSeed = 42
    if rdDistGeom.EmbedMolecule(m, p) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(m, maxIters=2000) if AllChem.MMFFHasAllMoleculeParams(m) else AllChem.UFFOptimizeMolecule(m)
    m.SetProp("_Name", cid)
    return Chem.MolToMolBlock(m)


def prep_crystal(out, manifest):
    w = Chem.SDWriter(out)
    for cid, v in manifest.items():
        m = Chem.AddHs(Chem.MolFromMolFile(v["ligand"]), addCoords=True)
        m.SetProp("_Name", cid)
        w.write(m)
    w.close()


def prep_rdkit(work, out, manifest, procs):
    with get_context("spawn").Pool(procs) as pool:
        blocks = pool.map(_rdkit_one, [(c, v["smiles"]) for c, v in manifest.items()])
    with open(out, "w") as f:
        for b in blocks:
            if b:
                f.write(b + "$$$$\n")


# ---------- geometry comparison ----------

def skeleton(mol):
    """Heavy-atom graph with bond orders, charges and aromaticity erased, so
    protomers and tautomers of the same compound map onto the crystal."""
    rw = Chem.RWMol(Chem.RemoveAllHs(mol, sanitize=False))
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


def matches(model_skel, crys_skel):
    if model_skel.GetNumAtoms() != crys_skel.GetNumAtoms():
        return []
    return model_skel.GetSubstructMatches(crys_skel, uniquify=False, useChirality=False, maxMatches=2000)


def kabsch_rmsd(p, q):
    p, q = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean()))


def ring_systems(crys):
    """Atom index lists of fused non-aromatic ring systems in the crystal ligand."""
    rings = [set(r) for r in crys.GetRingInfo().AtomRings()
             if not all(crys.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    systems = []
    for r in rings:
        merged = [s for s in systems if len(s & r) >= 2]
        for s in merged:
            systems.remove(s)
            r |= s
        systems.append(r)
    return [sorted(s) for s in systems]


def heavy_pos(mol):
    return Chem.RemoveAllHs(mol, sanitize=False).GetConformer().GetPositions()


def _signed_volume(pos, c, nb):
    return float(np.dot(pos[nb[0]] - pos[c], np.cross(pos[nb[1]] - pos[c], pos[nb[2]] - pos[c]))) > 0


def _is_cis(pos, c, a, b, d):
    n1, n2 = np.cross(pos[c] - pos[a], pos[b] - pos[a]), np.cross(pos[a] - pos[b], pos[d] - pos[b])
    return float(np.dot(n1, n2)) < 0  # dihedral c-a-b-d below 90 degrees


def stereo_elements(crys):
    """The crystal's stereo as geometry (signed volumes at tetrahedral centers,
    cis/trans across double bonds), so it can be checked on a protomer or
    tautomer through the skeleton mapping, where CIP labels may change."""
    m = Chem.Mol(crys)
    Chem.AssignStereochemistryFrom3D(m)
    pos = m.GetConformer().GetPositions()
    tet, dbl = [], []
    for idx, _ in Chem.FindMolChiralCenters(m, includeUnassigned=False, useLegacyImplementation=False):
        nb = [n.GetIdx() for n in m.GetAtomWithIdx(idx).GetNeighbors()][:3]
        if len(nb) == 3:
            tet.append((idx, nb, _signed_volume(pos, idx, nb)))
    for b in m.GetBonds():
        if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            a, z = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            c = next(n.GetIdx() for n in m.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() != z)
            d = next(n.GetIdx() for n in m.GetAtomWithIdx(z).GetNeighbors() if n.GetIdx() != a)
            dbl.append((c, a, z, d, _is_cis(pos, c, a, z, d)))
    return tet, dbl


def stereo_ok(mp, mt, elems):
    tet, dbl = elems
    return all(_signed_volume(mp, mt[c], [mt[n] for n in nb]) == v for c, nb, v in tet) and \
        all(_is_cis(mp, mt[c], mt[a], mt[z], mt[d]) == v for c, a, z, d, v in dbl)


def compare(model, crys, crys_skel, elems):
    """(RMSD with the crystal's stereo, RMSD ignoring stereo, stereo matches?);
    symmetry-corrected heavy-atom RMSD without superposition (docked pose)."""
    mp, cp = heavy_pos(model), crys.GetConformer().GetPositions()
    best_ok = best_any = math.inf
    for mt in matches(skeleton(model), crys_skel):
        r = float(np.sqrt(((mp[list(mt)] - cp) ** 2).sum(1).mean()))
        best_any = min(best_any, r)
        if stereo_ok(mp, mt, elems):
            best_ok = min(best_ok, r)
    return best_ok, best_any, best_ok < math.inf


def ring_rmsd(model, crys, crys_skel, systems, elems):
    """Worst ring system's superposed RMSD under the best stereo-consistent mapping."""
    mp, cp = heavy_pos(model), crys.GetConformer().GetPositions()
    best = math.inf
    for mt in matches(skeleton(model), crys_skel):
        if stereo_ok(mp, mt, elems):
            best = min(best, max(kabsch_rmsd(mp[[mt[i] for i in s]], cp[s]) for s in systems))
    return best


# ---------- per-model checks ----------

def assess(job):
    """job: (method, index, molblock, cid, ligand_path)"""
    import warnings

    warnings.filterwarnings("ignore")
    from posebusters import PoseBusters

    method, idx, block, cid, lig = job
    model = Chem.MolFromMolBlock(block, removeHs=False, sanitize=False)
    crys = Chem.MolFromMolFile(lig)
    cskel, elems = skeleton(crys), stereo_elements(crys)
    res = {"method": method, "cid": cid, "idx": idx, "graph_ok": bool(matches(skeleton(model), cskel))}
    res["stereo_ok"] = res["graph_ok"] and compare(model, crys, cskel, elems)[2]
    try:
        Chem.SanitizeMol(model)
        df = PoseBusters(config="mol").bust([model])
        res["valid"] = bool(df.iloc[0].all())
        res["failed_checks"] = [c for c in df.columns if not df.iloc[0][c]]
    except Exception as exc:
        res["valid"], res["failed_checks"] = False, [f"error: {exc}"[:80]]
    systems = ring_systems(crys)
    res["ring_rmsd"] = ring_rmsd(model, crys, cskel, systems, elems) if systems and res["graph_ok"] else None
    return res


# ---------- docking ----------

def _pose_record(method, idx, cid, seed, score, pose_path, crys, cskel, elems):
    pose = next(iter(Chem.SDMolSupplier(pose_path, removeHs=False, sanitize=False)), None) if pose_path else None
    if pose is None:
        return {"method": method, "idx": idx, "cid": cid, "seed": seed, "score": None, "rmsd": None, "rmsd_any": None}
    r_ok, r_any, _ = compare(pose, crys, cskel, elems)
    return {"method": method, "idx": idx, "cid": cid, "seed": seed, "score": score, "rmsd": r_ok, "rmsd_any": r_any}


def dock_smina(job):
    """job: (method, index, molblock, cid, ligand_path, protein_path, dock_dir, seed)"""
    method, idx, block, cid, lig, prot, ddir, seed = job
    tag = "" if seed == "1" else f"_s{seed}"  # seed 1 keeps the original cache names
    inp, out = os.path.join(ddir, f"{method}_{idx}_in.sdf"), os.path.join(ddir, f"{method}_{idx}{tag}_out.sdf")
    if not os.path.exists(out):
        with open(inp, "w") as f:
            f.write(block + "$$$$\n")
        subprocess.run([SMINA, "-r", prot, "-l", inp, "--autobox_ligand", lig, "--autobox_add", "8",
                        "--exhaustiveness", "8", "--num_modes", "9", "--seed", seed, "--cpu", "1", "-o", out, "-q"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
    first = next(iter(Chem.SDMolSupplier(out, removeHs=False, sanitize=False)), None) if os.path.exists(out) else None
    score = float(first.GetProp("minimizedAffinity")) if first is not None else None
    crys = Chem.MolFromMolFile(lig)
    return [_pose_record(method, idx, cid, seed, score, out if first is not None else None,
                         crys, skeleton(crys), stereo_elements(crys))]


def dock_aco(job):
    """job: (cid, ligand_path, protein_path, dock_dir, seeds, [(method, index, molblock)]).
    One helper process per complex, in the Meeko env; returns pose records,
    plus records with score None for models Meeko could not prepare."""
    cid, lig, prot, ddir, seeds, models = job
    out = os.path.join(ddir, "aco", cid)
    mdir = os.path.join(out, "models")
    os.makedirs(mdir, exist_ok=True)
    paths = []
    for method, idx, block in models:
        p = os.path.join(mdir, f"{method}_{idx}.sdf")
        if not os.path.exists(p):
            open(p, "w").write(block + "$$$$\n")
        paths.append(p)
    rpath = os.path.join(out, "results.json")
    if not os.path.exists(rpath):
        subprocess.run([ACO_PY, os.path.join(os.path.dirname(os.path.abspath(__file__)), "aco_dock.py"), prot, lig,
                        out, ",".join(seeds), *paths], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = json.load(open(rpath))
    crys = Chem.MolFromMolFile(lig)
    cskel, elems = skeleton(crys), stereo_elements(crys)
    recs = []
    for method, idx, _ in models:
        r = res[f"{method}_{idx}"]
        for seed in seeds:
            hit = r["seeds"].get(seed)
            rec = _pose_record(method, idx, cid, seed, hit and hit["score"], hit and hit["sdf"], crys, cskel, elems)
            rec["meeko_ok"] = r["meeko_ok"]
            recs.append(rec)
    return recs


# ---------- report ----------

def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = [x[rng.integers(len(x), size=len(x))].mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def report(work, manifest, assessed, docked, timings, methods, engine, seeds):
    cids = sorted(manifest)
    ring_cids = {c for c in cids if ring_systems(Chem.MolFromMolFile(manifest[c]["ligand"]))}
    per, success = {}, {}
    for m in methods:
        rows = [r for r in assessed if r["method"] == m]
        drows = [r for r in docked if r["method"] == m]
        # per complex: fraction of seeds whose top-scored pose (over all models) is <= 2 A with crystal stereo
        s_ok, s_any, s_oracle, seed_hits = {}, {}, {}, {seed: [] for seed in seeds}
        for c in cids:
            hits_ok, hits_any, hits_or = [], [], []
            for seed in seeds:
                rs = [r for r in drows if r["cid"] == c and r["seed"] == seed and r["score"] is not None]
                best = min(rs, key=lambda r: r["score"]) if rs else None
                hits_ok.append(bool(best and best["rmsd"] <= 2.0))
                seed_hits[seed].append(hits_ok[-1])
                hits_any.append(bool(best and best["rmsd_any"] <= 2.0))
                hits_or.append(any(r["rmsd"] <= 2.0 for r in rs))
            s_ok[c], s_any[c], s_oracle[c] = np.mean(hits_ok), np.mean(hits_any), np.mean(hits_or)
        by_seed = [np.mean(h) for h in seed_hits.values()]
        ring = [min([r["ring_rmsd"] for r in rows if r["cid"] == c and r["ring_rmsd"] is not None] or [math.inf])
                for c in ring_cids]
        fails = {}
        for r in rows:
            for k in r["failed_checks"]:
                fails[k] = fails.get(k, 0) + 1
        meeko_fail = sum(1 for r in drows if r.get("meeko_ok") is False) // max(1, len(seeds))
        success[m] = s_ok
        per[m] = {
            "models": len(rows), "with_models": len({r["cid"] for r in rows}), "complexes": len(cids),
            "valid": float(np.mean([r["valid"] for r in rows])) if rows else 0.0,
            "graph_ok": float(np.mean([r["graph_ok"] for r in rows])) if rows else 0.0,
            "stereo_models": float(np.mean([r["stereo_ok"] for r in rows])) if rows else 0.0,
            "stereo_recall": float(np.mean([any(r["stereo_ok"] for r in rows if r["cid"] == c) for c in cids])),
            "failed_checks": dict(sorted(fails.items(), key=lambda kv: -kv[1])[:6]),
            "ring_le_0.5": float(np.mean([x <= 0.5 for x in ring])) if ring else None,
            "ring_median": float(np.median(ring)) if ring else None, "ring_complexes": len(ring_cids),
            "meeko_failures": meeko_fail if engine == "aco" else None,
            "top1": float(np.mean(list(s_ok.values()))), "top1_ci": list(boot_ci(list(s_ok.values()))),
            "per_complex": {c: float(v) for c, v in s_ok.items()},
            "top1_seed_sd": float(np.std(by_seed)) if len(seeds) > 1 else None,
            "top1_ignoring_stereo": float(np.mean(list(s_any.values()))),
            "any_model": float(np.mean(list(s_oracle.values()))), "seconds_prep": timings.get(m),
        }
    lines = [f"Engine: {engine}, seeds {','.join(seeds)}; success = top-scored pose over all of a method's models "
             "within 2 A heavy-atom RMSD *with the crystal's stereochemistry*, averaged over seeds.", "",
             "| Method | Prep time | Models | PoseBusters-valid | Complexes with a crystal-stereo model | "
             "Ring systems <=0.5 A | Meeko failures | Top-1 success (95% CI) | seed SD | Top-1 ignoring stereo | Any model |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, p in per.items():
        ring = f"{p['ring_le_0.5']:.0%} ({p['ring_median']:.2f} A)" if p["ring_le_0.5"] is not None else "-"
        t = f"{p['seconds_prep']:.0f} s" if p["seconds_prep"] else "-"
        sd = f"{p['top1_seed_sd']:.1%}" if p["top1_seed_sd"] is not None else "-"
        mk = "-" if p["meeko_failures"] is None else str(p["meeko_failures"])
        lines.append(f"| {m} | {t} | {p['models']} | {p['valid']:.1%} | {p['stereo_recall']:.0%} | {ring} | {mk} | "
                     f"{p['top1']:.1%} ({p['top1_ci'][0]:.0%}-{p['top1_ci'][1]:.0%}) | {sd} | "
                     f"{p['top1_ignoring_stereo']:.1%} | {p['any_model']:.1%} |")
    if "vsgen" in per:
        for other in [m for m in per if m != "vsgen"]:
            diff = [success["vsgen"][c] - success[other][c] for c in cids]
            lo, hi = boot_ci(diff)
            lines.append(f"\nvsgen - {other}: top-1 success {np.mean(diff):+.1%} (95% CI {lo:+.1%} to {hi:+.1%})")
    for m, p in per.items():
        lines.append(f"\n{m}: failed PoseBusters checks: {p['failed_checks'] or 'none'}; models on the crystal graph "
                     f"{p['graph_ok']:.1%}, with crystal stereo {p['stereo_models']:.1%}")
    text = "\n".join(lines)
    open(os.path.join(work, f"report_{engine}.md"), "w").write(text + "\n")
    json.dump(per, open(os.path.join(work, f"summary_{engine}.json"), "w"), indent=1)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--methods", default="crystal,vsgen,gypsum,rdkit")  # also: molscrub
    ap.add_argument("--engine", choices=["smina", "aco"], default="smina")
    ap.add_argument("--seeds", default="1", help="docking seeds, e.g. 1,2,3")
    ap.add_argument("--strip-stereo", action="store_true", help="drop stereo from input SMILES (tests enumeration)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--ids", help="file with complex IDs to keep")
    ap.add_argument("--paired-with", help="another WORKDIR on the same complexes: paired top-1 differences per method")
    ap.add_argument("--gpu", choices=["off", "auto", "on"], default="off",
                    help="vsgen GPU use; off makes prep time CPU-vs-CPU with Gypsum-DL")
    a = ap.parse_args()
    seeds = a.seeds.split(",")
    os.makedirs(os.path.join(a.work, "dock"), exist_ok=True)
    ids = set(open(a.ids).read().split()) if a.ids else None
    tpath = os.path.join(a.work, "timings.json")
    timings = json.load(open(tpath)) if os.path.exists(tpath) else {}
    configs = timings.setdefault("config", {})
    shaping = f"{sorted(map(os.path.abspath, a.datasets))} strip_stereo={a.strip_stereo} limit={a.limit} ids={a.ids}"
    if configs.setdefault("inputs", shaping) != shaping:
        raise SystemExit(f"{a.work} holds inputs made with {configs['inputs']}; use a new WORKDIR")
    manifest = build_inputs(a.work, a.datasets, a.limit, ids, a.strip_stereo)
    inputs_hash = hashlib.sha256(open(os.path.join(a.work, "inputs.smi"), "rb").read()).hexdigest()[:12]
    methods = a.methods.split(",")
    models = []  # (method, index, molblock, cid)
    for m in methods:
        out = os.path.join(a.work, f"{m}.sdf")
        cfg = f"{vsgen_config(a.gpu) if m == 'vsgen' else m} inputs {inputs_hash}"
        if os.path.exists(out) and configs.get(m) != cfg:
            raise SystemExit(f"{out} was made with other settings ({configs.get(m)}); delete it to rebuild")
        if not os.path.exists(out):
            t = time.perf_counter()
            if m == "vsgen":
                prep_vsgen(a.work, out, a.gpu)
            elif m == "gypsum":
                prep_gypsum(a.work, out)
            elif m == "crystal":
                prep_crystal(out, manifest)
            elif m == "molscrub":
                prep_molscrub(a.work, out, a.procs)
            else:
                prep_rdkit(a.work, out, manifest, a.procs)
            timings[m] = time.perf_counter() - t
            configs[m] = cfg
            json.dump(timings, open(tpath, "w"))
        for i, mol in enumerate(Chem.SDMolSupplier(out, removeHs=False, sanitize=False)):
            if mol is not None and mol.GetProp("_Name") in manifest:
                models.append((m, i, Chem.MolToMolBlock(mol, kekulize=False), mol.GetProp("_Name")))
    print(f"{len(manifest)} complexes, {len(models)} models, engine {a.engine}, seeds {a.seeds}", flush=True)
    ddir = os.path.join(a.work, "dock")
    with get_context("spawn").Pool(a.procs) as pool:
        assessed = pool.map(assess, [(m, i, b, c, manifest[c]["ligand"]) for m, i, b, c in models], chunksize=4)
        if a.engine == "smina":
            jobs = [(m, i, b, c, manifest[c]["ligand"], manifest[c]["protein"], ddir, s)
                    for m, i, b, c in models for s in seeds]
            docked = [r for rs in pool.imap_unordered(dock_smina, jobs, chunksize=1) for r in rs]
        else:
            jobs = [(c, v["ligand"], v["protein"], ddir, seeds, [(m, i, b) for m, i, b, cc in models if cc == c])
                    for c, v in manifest.items()]
            docked = [r for rs in pool.imap_unordered(dock_aco, jobs, chunksize=1) for r in rs]
    json.dump({"assessed": assessed, "docked": docked}, open(os.path.join(a.work, f"results_{a.engine}.json"), "w"))
    report(a.work, manifest, assessed, docked, timings, methods, a.engine, seeds)
    if a.paired_with:
        paired(a.work, a.paired_with, a.engine)


def paired(work, other, engine):
    """This run minus another, per method, paired over complexes (e.g. one setting changed)."""
    mine, theirs = (json.load(open(os.path.join(w, f"summary_{engine}.json"))) for w in (work, other))
    if set(mine) != set(theirs):
        raise SystemExit(f"methods differ: {sorted(mine)} vs {sorted(theirs)}")
    for m in mine:
        a, b = mine[m]["per_complex"], theirs[m]["per_complex"]
        if set(a) != set(b):
            raise SystemExit(f"{m}: the two runs cover different complexes")
        cids = sorted(a)
        diff = [a[c] - b[c] for c in cids]
        lo, hi = boot_ci(diff)
        print(f"{m}: {os.path.basename(work)} - {os.path.basename(other)} top-1 {np.mean(diff):+.1%} "
              f"(95% CI {lo:+.1%} to {hi:+.1%}); better on {sum(d > 0 for d in diff)}, "
              f"worse on {sum(d < 0 for d in diff)} of {len(cids)} complexes")


if __name__ == "__main__":
    main()
