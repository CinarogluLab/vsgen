"""`vsgen-protomers`: build a Uni-pKa protomer cache for `vsgen --protomers-from`.

Separate from `vsgen` on purpose: Uni-pKa is ~100x the cost of the default
protonation, so it is run once per library, is resumable, and never competes
with the 3D stage for the GPU.
"""

import argparse
import os
import sys
import time

from vsgen import pka_unipka
from vsgen.run import read_inputs


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="vsgen-protomers",
        description="Protonation microstates from Uni-pKa, cached for vsgen --protomers-from.",
    )
    ap.add_argument("input", help=".smi or .sdf, as for vsgen")
    ap.add_argument("-o", "--cache", required=True, help="JSONL cache; an existing one is extended, not redone")
    ap.add_argument("--checkpoint", default=os.environ.get("UNIPKA_CHECKPOINT", ""),
                    help="Uni-pKa checkpoint (default: $UNIPKA_CHECKPOINT); not redistributed, see README")
    ap.add_argument("--ph", type=float, default=7.4)
    ap.add_argument("--min-population", type=float, default=0.05, help="drop microstates below this fraction")
    ap.add_argument("--max-states", type=int, default=5)
    ap.add_argument("--batch", type=int, default=40, help="molecules per model call")
    ap.add_argument("--gpu", action="store_true", help="CUDA; do not run while vsgen uses the GPU")
    a = ap.parse_args(argv)
    if not a.checkpoint or not os.path.isfile(a.checkpoint):
        ap.error("--checkpoint/UNIPKA_CHECKPOINT must point to a Uni-pKa checkpoint file "
                 "(see README: 'Uni-pKa backend'); vsgen does not redistribute it")
    try:
        import unipkainfer  # noqa: F401
    except ImportError:
        ap.error('Uni-pKa support is optional: pip install "vsgen[unipka]"')
    inputs = [(name, smiles) for _, name, smiles, _ in read_inputs(a.input)]
    t = time.perf_counter()
    summary = pka_unipka.build_cache(inputs, a.cache, a.checkpoint, ph=a.ph, batch=a.batch, gpu=a.gpu,
                                     min_population=a.min_population, max_states=a.max_states, log=sys.stderr)
    print(f"done: {summary} in {time.perf_counter() - t:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
