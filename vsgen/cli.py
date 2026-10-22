import argparse

from vsgen import chem, conf
from vsgen.run import RunConfig, run


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="vsgen", description="Fast 3D ligand preparation for docking (a Gypsum-DL alternative)."
    )
    ap.add_argument("input", help=".smi (SMILES [name] per line) or .sdf")
    ap.add_argument("-o", "--output", required=True, help="output .sdf; failures go to <output>_failed.tsv")
    g = ap.add_argument_group("chemistry")
    g.add_argument("--ph-min", type=float, default=6.4)
    g.add_argument("--ph-max", type=float, default=8.4)
    g.add_argument("--pka-model", choices=["molgpka", "dimorphite"], default="molgpka",
                   help="per-molecule pKa (MolGpKa, default) or Dimorphite-DL substructure ranges")
    g.add_argument("--pka-margin", type=float, default=0.5, help="MolGpKa: enumerate sites this far outside the pH window")
    g.add_argument("--pka-precision", type=float, default=1.0, help="Dimorphite-DL only")
    g.add_argument("--protomers-from", default="", metavar="CACHE.jsonl",
                   help="use Uni-pKa microstates from vsgen-protomers instead of --pka-model (replaces tautomers too)")
    g.add_argument("--protomers-fallback", action="store_true",
                   help="with --protomers-from: prepare uncached inputs with --pka-model instead of failing them")
    g.add_argument("--max-variants", type=int, default=5, help="variants per input (default 5)")
    g.add_argument("--max-tautomers", type=int, default=4)
    g.add_argument("--no-tautomers", action="store_true")
    g.add_argument("--max-stereoisomers", type=int, default=4, help="per tautomer, for unassigned centers/bonds")
    g.add_argument("--durrant-filters", action="store_true", help="drop metals and Durrant-lab substructures")
    g = ap.add_argument_group("3D")
    g.add_argument("--starts", type=int, default=15, help="embedding starts per variant")
    g.add_argument("--thorough", action="store_true", help="fully minimize every start (slower)")
    g.add_argument("--ring-confs", type=int, default=2, help="max models per variant with distinct ring conformations")
    g.add_argument("--force-field", choices=["mmff94s", "uff"], default="mmff94s", help="minimization force field (MMFF94s falls back to UFF per molecule)")
    g = ap.add_argument_group("execution")
    g.add_argument("--procs", type=int, default=0, help="CPU worker processes (default: all cores)")
    g.add_argument("--gpu", choices=["auto", "on", "off"], default="auto", help="nvMolKit GPU worker")
    g.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    cfg = RunConfig(
        chemistry=chem.Options(
            ph_min=a.ph_min, ph_max=a.ph_max, pka_precision=a.pka_precision, max_variants=a.max_variants,
            pka_model=a.pka_model, pka_margin=a.pka_margin,
            tautomers=not a.no_tautomers, max_tautomers=a.max_tautomers,
            max_stereoisomers=a.max_stereoisomers, durrant_filters=a.durrant_filters, seed=a.seed,
        ),
        three_d=conf.Conf3D(starts=a.starts, thorough=a.thorough, ring_confs=a.ring_confs, force_field=a.force_field),
        procs=a.procs,
        gpu=a.gpu,
        protomers_from=a.protomers_from,
        protomers_fallback=a.protomers_fallback,
    )
    run(a.input, a.output, cfg)


if __name__ == "__main__":
    main()
