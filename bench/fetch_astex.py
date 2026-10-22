"""Fetch the Astex Diverse Set for bench/dockval.py.

85 protein-ligand crystal complexes, CC-BY-4.0, from the mirror published with
PoseBench (huggingface.co/datasets/BioinfoMachineLearning/astex_diverse_set).
Only each complex's crystal ligand and protein are kept (~33 MB of the 528 MB
archive). The archive is verified against its published SHA-256 before use.

usage: python bench/fetch_astex.py [outdir]     (default: bench/astex_diverse_set)
"""

import hashlib
import io
import os
import re
import sys
import tarfile
import urllib.request

URL = ("https://huggingface.co/datasets/BioinfoMachineLearning/astex_diverse_set/"
       "resolve/main/astex_diverse_set.tar.gz")
SHA256 = "c521c1cf5ef980211f44e60fa6d4e9f3e507dfad4957b682c7c2e16b16481fa6"
KEEP = re.compile(r"/([0-9A-Za-z]{4}_[0-9A-Za-z]+)/\1_(ligand\.sdf|protein\.pdb)$")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "astex_diverse_set")
    os.makedirs(out, exist_ok=True)
    print(f"downloading {URL}", flush=True)
    data = urllib.request.urlopen(URL, timeout=600).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        sys.exit(f"checksum mismatch: expected {SHA256}, got {digest}")
    kept = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar:
            name = member.name
            if not member.isfile() or not KEEP.search(name) or name.startswith("/") or ".." in name:
                continue
            complex_dir = os.path.basename(os.path.dirname(name))
            target = os.path.join(out, complex_dir)
            os.makedirs(target, exist_ok=True)
            with tar.extractfile(member) as src, open(os.path.join(target, os.path.basename(name)), "wb") as dst:
                dst.write(src.read())
            kept += 1
    print(f"{kept} files in {out} ({kept // 2} complexes)\n"
          f"Astex Diverse Set, CC-BY-4.0; cite the PoseBusters paper (doi:10.1039/D3SC04185A) when using it.")


if __name__ == "__main__":
    main()
