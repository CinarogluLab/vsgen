"""2D preparation: desalt -> protonate -> tautomers -> filters -> stereo -> cap.

Variant selection is a deterministic diversity round-robin, not Gypsum-DL's
raw-embedding energy culling (which is serial, stochastic and dominated run
time on realistic libraries).
"""

import hashlib
import itertools
import random
from dataclasses import dataclass

import numpy as np
from dimorphite_dl import protonate_smiles
from rdkit import Chem, RDLogger
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# Durrant-lab prohibited substructures, from Gypsum-DL (Apache-2.0,
# https://github.com/durrantlab/gypsum_dl, steps/smiles/DurrantLabFilter.py).
DURRANT_SMARTS = [
    "C=[N-]",
    "[N-]C=[N+]",
    "[nH+]c[n-]",
    "[#7+]~[#7+]",
    "[#7-]~[#7-]",
    "[!#7]~[#7+]~[#7-]~[!#7]",
    "[#5]",
    "O=[PH](=O)([#8])([#8])",
    "[#7]=C1[#7]=C[#7]C=C1",
    "N=c1cc[#7]c[#7]1",
    "[$(N)]=C[$([OH]),$([O-])]",
    "[$(N)]C(=C)[$([OH]),$([O-])]",
]
METALS = frozenset(
    Chem.GetPeriodicTable().GetAtomicNumber(s)
    for s in (
        "Li Na K Rb Cs Fr Be Mg Ca Sr Ba Ra Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
        "Hf Ta W Re Os Ir Pt Au Hg La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Ac Th Pa U "
        "Al Ga In Sn Tl Pb Bi"
    ).split()
)
_DURRANT = None


@dataclass(frozen=True)
class Options:
    ph_min: float = 6.4
    ph_max: float = 8.4
    pka_precision: float = 1.0  # Dimorphite-DL only
    pka_model: str = "molgpka"  # molgpka (per-molecule pKa) | dimorphite (substructure pKa ranges)
    pka_margin: float = 0.5  # molgpka: also enumerate sites this far outside the pH window (~model error)
    max_protomers: int = 5
    tautomers: bool = True
    max_tautomers: int = 4
    tautomer_score_window: int = 2
    tautomers_may_change_stereo: bool = False
    max_stereoisomers: int = 4
    max_variants: int = 5
    durrant_filters: bool = False
    seed: int = 42


@dataclass(frozen=True)
class Variant:
    input_index: int
    name: str
    input_smiles: str
    smiles: str  # canonical isomeric, no explicit H
    lineage: str  # "p<i>t<j>s<k>": protomer, tautomer, stereoisomer ranks
    seed: int
    stereo_undefined: int = 0  # stereo elements the input left unspecified
    stereo_total: int = 1  # stereoisomers those imply (2^undefined, capped at a large value)
    stereo_emitted: int = 1  # distinct stereoisomers this input actually got


def variant_seed(global_seed, smiles):
    """Per-chemistry seed: the same structure gets the same seed in any run order."""
    digest = hashlib.blake2s(f"{global_seed}|{smiles}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def canon(mol):
    return Chem.MolToSmiles(mol)


def _nonaromatic_rings(mol):
    return sum(1 for r in mol.GetRingInfo().AtomRings() if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))


def _stereo_counts(mol):
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    return len(centers), sum(1 for _, label in centers if label != "?")


def desalt(mol):
    return rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)


# pKa shift that a charged site imposes on a same-type site n bonds away
# (textbook diamines/diacids: piperazine 9.7/5.4, ethylenediamine 9.9/6.9,
# 1,3-diaminopropane 10.6/8.6, malonic 2.8/5.7, succinic 4.2/5.6).
_COUPLING = {1: 3.5, 2: 3.5, 3: 3.5, 4: 2.5, 5: 1.5, 6: 1.2, 7: 0.9, 8: 0.9}
MAX_UNCERTAIN_SITES = 6  # 2^6 combinations; max_variants truncates long before this
_FULL_ENUMERATION_LIMIT = 64  # stereoisomers enumerated in full before sampling instead


def _molgpka_protomers(smiles, o):
    """Microstates at pH (ph_min+ph_max)/2 from MolGpKa per-site pKa.

    A site is charged when its pKa is beyond the window on the charged side,
    enumerated when inside it, neutral otherwise. Sites are charged strongest
    first, and each charged site shifts same-type sites nearby (_COUPLING).
    Enumerated states are ranked by their population (independent sites).
    """
    from vsgen import pka

    mol = rdMolStandardize.Uncharger().uncharge(Chem.MolFromSmiles(smiles))
    _, acids, bases = pka.predict(mol)
    return states_from_sites(mol, acids, bases, o)


def states_from_sites(mol, acids, bases, o, couple=True):
    """Protomers of `mol` (neutral) from per-site pKa: {heavy atom index: pKa}
    for acids and bases. couple=False when the pKa already account for
    neighboring charges (sequential predictors)."""
    dist = Chem.GetDistanceMatrix(mol)
    ph = (o.ph_min + o.ph_max) / 2
    todo = [(i, p, -1) for i, p in acids.items()] + [(i, p, +1) for i, p in bases.items()]
    charged, uncertain = [], []

    def effective(site):
        i, p, q = site
        shift = couple * sum(_COUPLING.get(int(dist[i][j]), 0.6) for j, _, qq in charged if qq == q and j != i)
        return p - shift if q > 0 else p + shift

    lo, hi = o.ph_min - o.pka_margin, o.ph_max + o.pka_margin  # widened by the predictor's uncertainty
    while todo:  # drive = how far beyond the window the site is, on the charged side
        drive = [(effective(t) - hi) if t[2] > 0 else (lo - effective(t)) for t in todo]
        k = int(np.argmax(drive))
        if drive[k] <= 0:
            break
        charged.append(todo.pop(k))
    uncertain = [(t[0], effective(t), t[2]) for t in todo if lo <= effective(t) <= hi]
    # Nearest the window first, so a molecule with many uncertain sites keeps the
    # ones whose state is genuinely in doubt rather than the lowest atom indices.
    uncertain.sort(key=lambda t: abs(t[1] - ph))
    uncertain = uncertain[:MAX_UNCERTAIN_SITES]

    def fraction_charged(site, pka, q, others):
        """Charged fraction of a site, shifted by the same-sign sites charged
        with it: coupling applies between uncertain sites too, not only against
        the forced ones, or a diamine's doubly charged state is overrated."""
        shift = couple * sum(_COUPLING.get(int(dist[site][j]), 0.6) for j in others if j != site)
        eff = pka - shift if q > 0 else pka + shift
        return 1 / (1 + 10 ** ((ph - eff) * q))

    states = []
    for bits in itertools.product((0, 1), repeat=len(uncertain)):
        on = {q: [i for (i, _, qq), b in zip(uncertain, bits) if b and qq == q] for q in (-1, 1)}
        pop = 1.0
        for (i, pka, q), b in zip(uncertain, bits):
            f = fraction_charged(i, pka, q, on[q])
            pop *= f if b else 1 - f
        states.append((pop, [(i, q) for (i, _, q), b in zip(uncertain, bits) if b]))
    states.sort(key=lambda s: -s[0])
    out, seen = [], set()
    for _, extra in states:
        rw = Chem.RWMol(mol)
        for i, q in [(i, q) for i, _, q in charged] + extra:
            atom = rw.GetAtomWithIdx(i)
            atom.SetNumExplicitHs(max(0, atom.GetTotalNumHs() + q))
            atom.SetNoImplicit(True)
            atom.SetFormalCharge(atom.GetFormalCharge() + q)
        try:
            m = rw.GetMol()
            Chem.SanitizeMol(m)
        except Exception:
            continue
        if canon(m) not in seen:
            seen.add(canon(m))
            out.append(m)
        if len(out) >= o.max_protomers:
            break
    return out


def protomers(smiles, o):
    # MolGpKa is out of its training domain below 6 heavy atoms (misses the dominant
    # state of e.g. ethylamine; 84.5% vs Dimorphite's 100% on IUPAC 3-5 atom molecules).
    if o.pka_model == "molgpka" and Chem.MolFromSmiles(smiles).GetNumHeavyAtoms() >= 6:
        try:
            out = _molgpka_protomers(smiles, o)
            if out:
                return out
        except Exception:
            pass  # fall back to Dimorphite-DL for molecules the predictor cannot handle
    try:
        out = protonate_smiles(
            smiles, ph_min=o.ph_min, ph_max=o.ph_max, precision=o.pka_precision, max_variants=o.max_protomers
        )
    except Exception:
        out = []
    seen, mols = set(), []
    for s in out or [smiles]:
        m = Chem.MolFromSmiles(s)
        if m is not None and canon(m) not in seen:
            seen.add(canon(m))
            mols.append(m)
    return mols or [Chem.MolFromSmiles(smiles)]


def tautomers(mol, o):
    """Source form first, then highest-scoring tautomers that keep the
    non-aromatic ring count and stereocenters (Gypsum-DL's artifact filters)."""
    if not o.tautomers or o.max_tautomers <= 1:
        return [mol]
    params = rdMolStandardize.CleanupParameters()
    params.maxTautomers = 64
    params.maxTransforms = 100
    params.tautomerRemoveSp3Stereo = False
    params.tautomerRemoveBondStereo = False
    enum = rdMolStandardize.TautomerEnumerator(params)
    try:
        cands = list(enum.Enumerate(mol))
    except Exception:
        return [mol]
    rings, (n_centers, n_assigned) = _nonaromatic_rings(mol), _stereo_counts(mol)
    src = canon(mol)
    # A tautomer may not introduce a Durrant-lab artifact (iminols, odd
    # adenine forms...) that the source form does not already have.
    src_hits = {i for i, q in enumerate(_durrant_queries()) if mol.HasSubstructMatch(q)}
    keep = []
    for t in cands:
        try:
            Chem.SanitizeMol(t)
        except Exception:
            continue
        if canon(t) == src or _nonaromatic_rings(t) != rings:
            continue
        if any(i not in src_hits and t.HasSubstructMatch(q) for i, q in enumerate(_durrant_queries())):
            continue
        if not o.tautomers_may_change_stereo and _stereo_counts(t) != (n_centers, n_assigned):
            continue
        keep.append(t)
    # RDKit's tautomer score rewards C=O, aromaticity etc.; implausible forms
    # (ester enols, enediolates) sit >=3 below the best, plausible ones within ~2.
    scores = {canon(t): enum.ScoreTautomer(t) for t in keep}
    best = max([enum.ScoreTautomer(mol), *scores.values()])
    keep = [t for t in keep if scores[canon(t)] >= best - o.tautomer_score_window]
    keep.sort(key=lambda t: (-scores[canon(t)], canon(t)))
    return [mol] + keep[: o.max_tautomers - 1]


def _durrant_queries():
    global _DURRANT
    if _DURRANT is None:
        _DURRANT = [Chem.MolFromSmarts(s) for s in DURRANT_SMARTS]
    return _DURRANT


def passes_durrant(mol):
    if any(a.GetAtomicNum() in METALS for a in mol.GetAtoms()):
        return False
    return not any(mol.HasSubstructMatch(q) for q in _durrant_queries())


def undefined_stereo(mol):
    """Number of stereo elements the input leaves unspecified."""
    try:
        return sum(1 for e in Chem.FindPotentialStereo(mol)
                   if e.specified == Chem.StereoSpecified.Unspecified)
    except Exception:
        return 0


def _descriptors(mol):
    """Stereo assignment of a molecule as a comparable tuple."""
    centers = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))
    # any non-NONE stereo: the enumerator marks bonds STEREOE/Z or STEREOCIS/TRANS
    # depending on the molecule, and missing them would hide E/Z from the diversity pick
    bonds = {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()): str(b.GetStereo()) for b in mol.GetBonds()
             if b.GetStereo() != Chem.BondStereo.STEREONONE}
    return tuple(centers.get(i) for i in sorted(centers)) + tuple(bonds[k] for k in sorted(bonds))


def _diverse_subset(isomers, k):
    """k isomers that differ in as many stereo elements as possible: picking
    RDKit's first k leaves neighbours that differ at one centre, which is the
    worst use of a small budget. Deterministic (canonical SMILES order)."""
    if len(isomers) <= k:
        return isomers
    coded = [(canon(m), _descriptors(m), m) for m in isomers]
    coded.sort(key=lambda x: x[0])
    chosen = [coded[0]]
    while len(chosen) < k:
        far = max(coded, key=lambda c: (min(sum(a != b for a, b in zip(c[1], p[1])) for p in chosen), -1)
                  if c not in chosen else (-1, 0))
        chosen.append(far)
    return [c[2] for c in chosen]


def _shift_h(mol, atoms, q):
    """Add (q=+1) or remove (q=-1) one H+ on each atom; a removal clears the atom's stereo tag."""
    m = Chem.RWMol(mol)
    for i in atoms:
        a = m.GetAtomWithIdx(i)
        a.SetFormalCharge(a.GetFormalCharge() + q)
        a.SetNumExplicitHs(a.GetTotalNumHs() + q)
        a.SetNoImplicit(True)
        if q < 0:
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    Chem.SanitizeMol(m)
    return m.GetMol()


def _labile_nh(mol):
    """[N+]-H atoms whose configuration is not stable: proton exchange inverts
    them unless the neutral amine's N is itself a stereocentre (aziridines,
    Troger's base), so their R/S is not a stereoisomer."""
    cands = [a.GetIdx() for a in mol.GetAtoms()
             if a.GetAtomicNum() == 7 and a.GetFormalCharge() == 1 and a.GetTotalNumHs() > 0]
    if not cands:
        return []
    try:
        real = {e.centeredOn for e in Chem.FindPotentialStereo(_shift_h(mol, cands, -1))}
    except Exception:
        return []
    return [i for i in cands if i not in real]


def stereoisomers(mol, o, seed):
    # Enumerate on the conjugate base of invertible R3NH+, so neither the N nor
    # the centres it would induce (C4 of a piperidinium) take stereo slots;
    # re-protonate afterwards (same atom order).
    labile = _labile_nh(mol)
    work = _shift_h(mol, labile, -1) if labile else mol
    n = undefined_stereo(work)
    # Enumerate the whole set when it is small, so the subset can be chosen for
    # diversity; fall back to RDKit's seeded sample when 2^n is too large.
    full = n and 2 ** n <= _FULL_ENUMERATION_LIMIT
    opts = StereoEnumerationOptions(
        onlyUnassigned=True, unique=True, rand=random.Random(seed),
        maxIsomers=0 if full else o.max_stereoisomers,
    )
    try:
        isos = list(EnumerateStereoisomers(work, options=opts))
    except Exception:
        isos = [work]
    if labile:
        isos = [_shift_h(m, labile, +1) for m in isos]
    isos = sorted({canon(m): m for m in isos}.values(), key=canon) or [mol]
    return _diverse_subset(isos, o.max_stereoisomers) if full else isos


def cache_key(smiles):
    """Key a protomer cache by: canonical SMILES of the desalted input."""
    mol = Chem.MolFromSmiles(smiles)
    return canon(desalt(mol)) if mol is not None else ""


def prepare_2d(input_index, name, smiles, o=Options(), states=None):
    """Return (variants, error). Variants are capped and ordered by a
    round-robin over depth: every protomer's base form first, then alternates.

    `states`: microstate SMILES for this input (a Uni-pKa cache entry), used
    instead of the protomer and tautomer stages, whose job they already do.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], "unparseable SMILES"
    mol = desalt(mol)
    if mol.GetNumAtoms() == 0:
        return [], "empty after desalting"
    found = {}  # canonical smiles -> (rank key, lineage)
    forms = [[Chem.MolFromSmiles(x)] for x in states] if states else \
        [tautomers(pm, o) for pm in protomers(canon(mol), o)]
    undefined = undefined_stereo(mol)
    for p, taut in enumerate(forms):
        for t, tm in enumerate(taut):
            if tm is None:
                continue
            if o.durrant_filters and not passes_durrant(tm):
                continue
            for s, sm in enumerate(stereoisomers(tm, o, variant_seed(o.seed, canon(tm)))):
                key = canon(sm)
                # Spend the budget on what the input leaves undefined: the dominant
                # form's stereoisomers come before alternative protomers, which come
                # before alternative tautomers. Measured on stereo-stripped Astex:
                # the old depth-first order let alternative protomers take the slots.
                tier = 0 if (p == 0 and t == 0) else (1 if t == 0 else 2)
                rank = (tier, s, p, t)
                if key not in found or rank < found[key][0]:
                    found[key] = (rank, f"p{p}t{t}s{s}")
    if not found:
        return [], "no variant passed filters"
    chosen = sorted(found.items(), key=lambda kv: kv[1][0])[: o.max_variants]
    emitted = len({_descriptors(Chem.MolFromSmiles(smi)) for smi, _ in chosen})
    total = 2 ** undefined if undefined < 20 else -1  # -1: too many to count
    return [
        Variant(input_index, name, smiles, smi, lineage, variant_seed(o.seed, smi), undefined, total, emitted)
        for smi, (_, lineage) in chosen
    ], None
