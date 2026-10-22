"""Convert the official Uni-pKa checkpoint (t_dwar_v_novartis_a_b.pt, from the
authors' Bohrium notebook) to unipkainfer's key layout, tensors only.
Keys get the "unimol." prefix and head "chembl_all" becomes "chembl"; the
pickle holds only tensors, NumPy scalars/dtypes, Namespace and OrderedDict.

usage: python bench/convert_unipka_official.py t_dwar_v_novartis_a_b.pt out/checkpoint_best.pt
"""
import _codecs
import argparse
import os
import sys

import numpy as np
import torch

dtypes = [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith("DType")]
safe = [argparse.Namespace, _codecs.encode, np.dtype, np._core.multiarray.scalar,
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), *dtypes]
with torch.serialization.safe_globals(safe):
    model = torch.load(sys.argv[1], map_location="cpu", weights_only=True)["model"]
os.makedirs(os.path.dirname(os.path.abspath(sys.argv[2])), exist_ok=True)
torch.save({"model": {"unimol." + k.replace("classification_heads.chembl_all.", "classification_heads.chembl."): v
                      for k, v in model.items()}}, sys.argv[2])
print(f"wrote {len(model)} tensors to {sys.argv[2]}")
