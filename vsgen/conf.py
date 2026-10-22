"""3D: ETKDGv3 embedding + staged force-field minimization, on CPU (RDKit) or GPU (nvMolKit).

Force field: UFF, or MMFF94s with a distance-dependent dielectric (eps = 4r),
an empirical screening (not a solvent model) so charged groups do not collapse
onto each other as in vacuum; molecules MMFF cannot type fall back to UFF.
Fast mode: embed N starts, 100 minimization steps on all, full minimization (2000) on
the best 3, keep the lowest; plus up to (ring_confs - 1) extra models whose
non-aromatic ring geometry differs. Thorough mode fully minimizes every start.
Embedding is capped at 50 iterations so impossible stereoisomers fail in
seconds instead of minutes (a cap that changed nothing on normal molecules).
"""

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDistGeom, rdMolAlign
from rdkit.Chem.rdForceFieldHelpers import OptimizeMoleculeConfs, UFFHasAllMoleculeParams

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class Conf3D:
    starts: int = 15
    max_embed_iters: int = 50
    stage_steps: int = 100
    full_steps: int = 2000
    keep_top: int = 3
    thorough: bool = False
    ring_confs: int = 2
    ring_rmsd: float = 0.3
    force_field: str = "mmff94s"  # mmff94s (falls back to UFF per molecule) | uff


def mmff_props(mol):
    """MMFF94s properties with a distance-dependent dielectric (eps = 4r), or None."""
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    if props is None:
        return None
    props.SetMMFFDielectricModel(2)  # RDKit: 1 = constant, 2 = distance-dependent
    props.SetMMFFDielectricConstant(4.0)
    return props


def _field(mol, cfg):
    """(name, props) of the force field used for this molecule; props None for UFF.
    name is None when no force field can type the molecule."""
    if cfg.force_field == "mmff94s":
        props = mmff_props(mol)
        if props is not None:
            return "MMFF94s", props
    return ("UFF", None) if UFFHasAllMoleculeParams(mol) else (None, None)


def _ff(mol, props, conf_id):
    if props is not None:
        return AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    return AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)


def embed_params(seed, cfg):
    p = rdDistGeom.ETKDGv3()
    p.enforceChirality = True
    p.useRandomCoords = True
    p.randomSeed = seed
    p.maxIterations = cfg.max_embed_iters
    p.numThreads = 1
    return p


def _stereo(mol):
    centers = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False))
    bonds = {
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx()): b.GetStereo()
        for b in mol.GetBonds()
        if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ)
    }
    return centers, bonds


def stereo_preserved(ref_noh, model):
    """Assigned centers and E/Z of the 2D variant survive in the 3D model."""
    noh = Chem.RemoveHs(model)
    Chem.AssignStereochemistryFrom3D(noh)
    (rc, rb), (gc, gb) = _stereo(ref_noh), _stereo(noh)
    return all(gc.get(i) == v for i, v in rc.items()) and all(gb.get(k) == v for k, v in rb.items())


def _ring_atom_sets(mol):
    ri = mol.GetRingInfo()
    return [list(r) for r in ri.AtomRings() if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]


def _ring_differs(mol, cid_a, cid_b, rings, cutoff):
    for ring in rings:
        rmsd = rdMolAlign.AlignMol(mol, mol, prbCid=cid_a, refCid=cid_b, atomMap=[(i, i) for i in ring])
        if rmsd > cutoff:
            return True
    return False


def select_models(smiles, mol, energies, cfg, field=None):
    """mol: H-added, with conformers aligned to `energies` (inf = not a candidate).
    Returns [(Chem.Mol single-conformer, energy, kind)]."""
    ref = Chem.MolFromSmiles(smiles)
    ids = [c.GetId() for c in mol.GetConformers()]
    order = [j for j in np.argsort(energies) if np.isfinite(energies[j])]
    if not order:  # no force field (e.g. unsupported atoms): keep raw embeddings
        order = list(range(len(ids)))
    field = field or "none"  # no force field: raw ETKDG coordinates, flagged so users can filter them
    rings = _ring_atom_sets(mol) if cfg.ring_confs > 1 else []
    picked = []
    for j in order:
        model = Chem.Mol(mol, confId=ids[j])
        model.GetConformer().SetId(0)
        model.SetProp("force_field", field)
        if not stereo_preserved(ref, model):
            continue
        if picked and not (rings and all(_ring_differs(mol, ids[j], ids[k], rings, cfg.ring_rmsd) for k, _ in picked)):
            continue
        picked.append((j, model))
        if len(picked) >= max(1, cfg.ring_confs if rings else 1):
            break
    e = lambda j: float(energies[j]) if np.isfinite(energies[j]) else None
    return [(m, e(j), "best" if n == 0 else "ring_alt") for n, (j, m) in enumerate(picked)]


def _cpu_minimize(mol, cfg):
    name, props = _field(mol, cfg)
    if name is None:
        return [float("inf")] * mol.GetNumConformers(), None
    ids = [c.GetId() for c in mol.GetConformers()]
    ff = _ff(mol, props, ids[0])
    if cfg.thorough:
        return [e for _, e in OptimizeMoleculeConfs(mol, ff, numThreads=1, maxIters=cfg.full_steps)], name
    staged = [e for _, e in OptimizeMoleculeConfs(mol, ff, numThreads=1, maxIters=cfg.stage_steps)]
    energies = [float("inf")] * len(staged)
    for j in np.argsort(staged)[: cfg.keep_top]:
        ff = _ff(mol, props, ids[j])
        ff.Minimize(maxIts=cfg.full_steps)
        energies[j] = ff.CalcEnergy()
    return energies, name


def _embed_cpu(mol, seed, cfg):
    """Probe with one start before paying for all of them: an impossible
    stereoisomer fails the probes in ~2 s instead of ~23 s for 15 capped starts."""
    for cap in (10, cfg.max_embed_iters):
        p = embed_params(seed, cfg)
        p.maxIterations = cap
        if rdDistGeom.EmbedMultipleConfs(mol, 1, p):
            break
    else:
        return False
    if cfg.starts > 1:
        p = embed_params(seed + 1, cfg)
        p.clearConfs = False
        rdDistGeom.EmbedMultipleConfs(mol, cfg.starts - 1, p)
    return True


def build_cpu(smiles, seed, cfg=Conf3D()):
    """One variant on CPU. Returns (models, error)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if not _embed_cpu(mol, seed, cfg):
        return [], "ETKDG embedding failed within the iteration cap"
    energies, field = _cpu_minimize(mol, cfg)
    models = select_models(smiles, mol, energies, cfg, field)
    return (models, None) if models else ([], "no conformer preserved the assigned stereochemistry")


def gpu_supported(smiles, cfg=Conf3D()):
    """The GPU batch uses one force field; molecules that would fall back go to the CPU."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    want = "MMFF94s" if cfg.force_field == "mmff94s" else "UFF"
    return _field(mol, cfg)[0] == want and mol.GetNumAtoms() <= 256


def build_gpu_batch(items, cfg=Conf3D(), batch_size=500):
    """items: [(smiles, seed)]. Same algorithm as build_cpu, batched on GPU.
    nvMolKit takes one seed per call (first item's), so coordinates are
    reproducible for a given batch composition, not per variant.
    Returns [(models, error) | None]; None = hand back to the CPU."""
    from nvmolkit.embedMolecules import EmbedMolecules
    from nvmolkit.types import HardwareOptions
    from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
    from nvmolkit.uffOptimization import UFFOptimizeMoleculesConfs

    hw = HardwareOptions()
    hw.batchSize = batch_size  # small batches were fastest on an 8 GB laptop GPU
    hw.batchesPerGpu = 1
    mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s, _ in items]
    EmbedMolecules(mols, embed_params(items[0][1], cfg), confsPerMolecule=cfg.starts,
                   maxIterations=cfg.max_embed_iters, hardwareOptions=hw)
    ok = [i for i, m in enumerate(mols) if m.GetNumConformers() > 0]
    mmff = cfg.force_field == "mmff94s"
    field = "MMFF94s" if mmff else "UFF"

    def optimize(ms, iters):
        if mmff:
            return MMFFOptimizeMoleculesConfs(ms, maxIters=iters, properties=[mmff_props(m) for m in ms])
        return UFFOptimizeMoleculesConfs(ms, maxIters=iters)

    energies = {}
    if ok:
        first = optimize([mols[i] for i in ok], cfg.full_steps if cfg.thorough else cfg.stage_steps)
        if cfg.thorough:
            energies = {i: list(e) for i, e in zip(ok, first)}
        else:
            # Full minimization of each molecule's best keep_top starts, as one batch.
            subs, keep_ids = [], []
            for i, e in zip(ok, first):
                ids = [c.GetId() for c in mols[i].GetConformers()]
                top = list(np.argsort(e)[: cfg.keep_top])
                sub = Chem.Mol(mols[i])
                sub.RemoveAllConformers()
                for j in top:
                    sub.AddConformer(mols[i].GetConformer(ids[j]), assignId=True)
                subs.append(sub)
                keep_ids.append(top)
            second = optimize(subs, cfg.full_steps)
            for i, sub, top, e2 in zip(ok, subs, keep_ids, second):
                full = [float("inf")] * mols[i].GetNumConformers()
                ids = [c.GetId() for c in mols[i].GetConformers()]
                for slot, (j, conf) in enumerate(zip(top, sub.GetConformers())):
                    mols[i].GetConformer(ids[j]).SetPositions(conf.GetPositions())
                    full[j] = float(e2[slot])
                energies[i] = full
    out = []
    for i, (smiles, _) in enumerate(items):
        if i not in energies:
            out.append(None)  # 0 conformers: let the CPU path try (and report) it
            continue
        models = select_models(smiles, mols[i], energies[i], cfg, field)
        out.append((models, None) if models else None)
    return out
