# vsgen

Fast 3D ligand preparation for docking. Give it SMILES or SDF; it writes
protonated, tautomer- and stereo-enumerated 3D models as SDF for Vina, smina,
Vina-ACO or any other docking program.

vsgen does the job of [Gypsum-DL](https://github.com/durrantlab/gypsum_dl) about
10x faster on CPU and 18x faster with a GPU. Its models redock as well as
Gypsum-DL's, and it emits far fewer charge states that experiment rules out.

## Install

```bash
mamba create -n vsgen -c conda-forge python=3.12 rdkit numpy
mamba activate vsgen
pip install .                            # RDKit, NumPy, Dimorphite-DL 2.1.0
mamba install -c conda-forge nvmolkit    # optional: NVIDIA GPU backend
pip install ".[unipka]"                  # optional: Uni-pKa protonation
```

Python 3.10 or newer. The default pKa model ships with the package (11 MB, no
PyTorch).

## Use

```bash
vsgen library.smi -o prepared.sdf                  # uses a GPU if one is present
vsgen library.smi -o prepared.sdf --gpu off        # reproducible from --seed
vsgen library.smi -o prepared.sdf --max-variants 2 --no-tautomers   # cheaper, for huge libraries
```

Each `.smi` line is `SMILES [name]`; SDF input keeps its properties. Failed
inputs go to `prepared_failed.tsv` with a reason. Every model records its
variant `SMILES`, `lineage` (protomer, tautomer, stereoisomer), `energy`,
`force_field` and `seed`. Run `vsgen --help` for all options.

## How it works

1. **2D.** Keep the largest fragment, protonate for pH 7.4 (window 6.4-8.4)
   from per-atom pKa predicted by MolGpKa, enumerate tautomers and any
   stereocentres the input leaves open. At most 5 variants per input; open
   stereo gets the budget first.
2. **3D.** Embed 15 ETKDGv3 starts, minimize them with MMFF94s and a
   distance-dependent dielectric (eps = 4r), and keep the lowest-energy model
   whose stereochemistry survived, plus up to one alternative ring conformation.
3. **Scheduling.** CPU workers take the most expensive molecules first, an
   optional GPU worker takes batches of the cheapest, and memory stays flat
   because input streams through a bounded window.

| | Gypsum-DL 1.3.0 | vsgen |
|---|---|---|
| Protonation | Dimorphite-DL group pKa ranges | per-atom pKa (MolGpKa; Uni-pKa optional) |
| Force field | UFF | MMFF94s, eps = 4r; UFF fallback |
| Variant selection | energy culling on a single core | fixed ranking, no extra embedding |
| Hardware | CPU | CPU, or CPU + NVIDIA GPU |

## Results


**Speed**, 5,000 compounds from NCI and ChEMBL, each tool with its defaults,
one run each:

| Tool | Wall time | Speedup | Models | Peak memory (whole run) |
|---|---|---|---|---|
| Gypsum-DL, 16 processes | 94.1 min | 1x | 18,188 | 13.6 GB |
| vsgen, CPU | 8.9 min | 10.5x | 10,885 | 1.2 GB |
| vsgen, CPU + GPU | 5.3 min | 17.7x | 10,912 | 2.0 GB |

vsgen writes 40% fewer models, mostly because it emits fewer unlikely charge
states, so the table compares defaults rather than identical work.

**Redocking**, 85 Astex Diverse Set complexes, Vina-ACO, 5 seeds. A success is a
top-scored pose within 2 A of the crystal ligand with the crystal's
stereochemistry.

| Preparation | Models | PoseBusters-valid | Success, stereo given | Success, stereo removed |
|---|---|---|---|---|
| Crystal ligand + H (ceiling) | 85 | 97.6% | 70.1% | 70.8% |
| **vsgen** | 256 | 100% | **63.8%** | **43.3%** |
| Gypsum-DL | 385 | 100% | 62.1% | 39.3% |
| One RDKit conformer, no protonation | 85 | 100% | 57.2% | 31.5% |

vsgen and Gypsum-DL redock equally well within the uncertainty of 85 complexes
(vsgen minus Gypsum-DL: +1.6 points, 95% CI -7.1 to +10.8). Both beat naive
preparation. With stereo removed from the input, vsgen produces the crystal
stereoisomer for 87% of ligands.

**Protonation** against measured pKa: 1,453 IUPAC molecules with one ionizable
group that are absent from the pKaSolver, Uni-pKa and QupKake training sets,
plus the blind SAMPL6-8 challenges. An "unlikely" state is one the measured pKa
puts more than about 1 unit on the wrong side of pH 7.4.

| Model | MAE on SAMPL | Unlikely charge states emitted | Cost per molecule |
|---|---|---|---|
| **MolGpKa** (default) | 0.78 | 12.9% | ~5 ms |
| Uni-pKa (optional) | 0.64 | 11.2% | 0.1-1.1 s |
| Dimorphite-DL 2.0.2 (Gypsum-DL 1.3.0's) | 1.62 * | 57.7% | ~1 ms |
| Dimorphite-DL 2.1.0 | 1.79 * | 38.6% | ~1 ms |

\* Dimorphite-DL assigns states from class pKa ranges and predicts no pKa; the
MAE uses each class's mean.

**Geometry** against 492 small-molecule crystal structures from the
Crystallography Open Database: bond lengths off by 0.015 A and angles by
1.46 degrees on average; amides and esters keep the crystal's cis/trans
arrangement; 85% of molecules with a saturated ring have a model within 0.3 A
of the crystal's ring pucker.

## Uni-pKa backend (optional)

Use it on subsets where protonation decides the answer. It costs about 100x the
default, so it runs as a separate, resumable step:

```bash
pip install "vsgen[unipka]"
export UNIPKA_CHECKPOINT=/path/to/checkpoint.pt
vsgen-protomers library.smi -o protomers.jsonl
vsgen library.smi -o prepared.sdf --protomers-from protomers.jsonl
```

vsgen ships Uni-pKa's microstate enumerator but no weights, because the
checkpoint carries no stated license. Get one from
[Uni-pKa](https://github.com/dptech-corp/Uni-pKa).

## Limits

- Give vsgen defined stereochemistry when you have it. With 5 variants per
  input it misses the crystal stereoisomer for about 1 in 8 Astex ligands, and
  redocking drops from 64% to 43%.
- Protonation and tautomers are predictions. Tautomers are not re-ranked for
  pH, and acid-base coupling inside zwitterions is not modelled.
- Each model is one MMFF94s minimum. Docking re-samples rotatable bonds, but
  rings and amides stay as vsgen built them.
- Molecules no force field can type keep raw ETKDG coordinates and carry
  `force_field=none`.
- GPU runs are not bitwise reproducible; use `--gpu off` when that matters.

## Tests and benchmarks

```bash
pip install -e ".[test]" && pytest -q
```

`bench/` holds the validation, each script with its pass criterion:
`dockval.py` (redocking), `pka_eval.py` and `pka_external.py` (protonation),
`cod_geom.py` (geometry), `ionpair.py` (charged groups), `regress.py`
(minimization) and `speed.py` (throughput).

## License and credits

Apache-2.0. vsgen builds on [Gypsum-DL](https://github.com/durrantlab/gypsum_dl),
[MolGpKa](https://github.com/Xundrug/MolGpKa),
[Uni-pKa](https://github.com/dptech-corp/Uni-pKa),
[Dimorphite-DL](https://github.com/durrantlab/dimorphite_dl) and
[RDKit](https://www.rdkit.org/).
