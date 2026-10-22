"""One-off: convert MolGpKa's PyTorch weights to the NumPy archives vsgen ships.

MolGpKa (MIT): https://github.com/Xundrug/MolGpKa, models/weight_{acid,base}.pth
usage: python bench/convert_molgpka.py MOLGPKA_CHECKOUT  (needs torch; vsgen itself does not)
"""

import sys

import numpy as np
import torch

for kind in ("acid", "base"):
    sd = torch.load(f"{sys.argv[1]}/models/weight_{kind}.pth", map_location="cpu")
    np.savez_compressed(f"vsgen/data/molgpka_{kind}.npz", **{k: v.numpy() for k, v in sd.items() if not k.endswith("num_batches_tracked")})
    print(kind, {k: tuple(v.shape) for k, v in sd.items()})
