"""Library-scale speed: vsgen (CPU only, CPU + GPU) vs Gypsum-DL, each with its defaults.

Runs each tool on the same .smi file, one at a time (nothing else should run
on the machine), and records wall time, peak RSS of the largest child process,
inputs with at least one model, models written, software versions and
hardware in OUTDIR/speed.json. Tools already in speed.json are skipped, so
the 3 h Gypsum-DL run is not repeated by accident; --force reruns them.

usage: python bench/speed.py OUTDIR INPUT.smi [--tools vsgen-cpu,vsgen-gpu,gypsum] [--force]
"""

import argparse
import json
import os
import shutil
import subprocess
import time
import psutil
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
VSGEN = os.environ.get("VSGEN", "vsgen")
GYPSUM = os.environ.get("GYPSUM", "gypsum-dl")


def _sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _interpreter(exe):
    """Python behind a console script (its shebang), so versions come from the measured tool's env."""
    path = shutil.which(exe) or exe
    first = open(path).readline()
    return path, first[2:].strip() if first.startswith("#!") else None


def environment(tool):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exe, py = _interpreter(GYPSUM if tool == "gypsum" else VSGEN)
    pkgs = ["gypsum_dl", "dimorphite_dl", "rdkit"] if tool == "gypsum" else ["dimorphite_dl", "rdkit", "nvmolkit"]
    probe = "import sys; from importlib.metadata import version as v\nfor p in sys.argv[1:]:\n    try: print(p, v(p))\n    except Exception: print(p, None)"
    versions = dict(l.split(" ", 1) for l in (_sh([py, "-c", probe] + pkgs) or "").splitlines())
    env = {"executable": exe, "python": _sh([py, "-c", "import sys; print(sys.version.split()[0])"]), **versions,
           "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), None),
           "logical_cpus": os.cpu_count(), "gpu": _sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])}
    if tool != "gypsum":
        env["vsgen_commit"] = _sh(["git", "-C", root, "rev-parse", "--short", "HEAD"])
        env["vsgen_dirty"] = bool(_sh(["git", "-C", root, "status", "--porcelain", "--", "vsgen"]))
    return env


def _tree_pss(proc):
    """Summed PSS (bytes) of a process and its descendants: shared pages (libraries,
    copy-on-write after fork) are split between processes, not counted once per worker."""
    total = 0
    for p in [proc] + proc.children(recursive=True):
        try:
            total += p.memory_full_info().pss
        except psutil.Error:
            pass
    return total


def count(sdf, need_prop=None):
    names, models = set(), 0
    for m in Chem.SDMolSupplier(sdf, removeHs=False, sanitize=False):
        if m is None or (need_prop and not m.HasProp(need_prop)):
            continue
        models += 1
        names.add(m.GetProp("_Name"))
    return len(names), models


def run_tool(tool, smi, out):
    os.makedirs(out, exist_ok=True)
    if tool.startswith("vsgen"):
        sdf = os.path.join(out, f"{tool}.sdf")
        cmd = [VSGEN, smi, "-o", sdf, "--gpu", "off" if tool == "vsgen-cpu" else "on"]
    else:
        gdir = os.path.join(out, "gypsum_out")
        os.makedirs(gdir, exist_ok=True)
        sdf = os.path.join(gdir, "gypsum_dl_success.sdf")
        cmd = [GYPSUM, "--source", smi, "--output_folder", gdir, "--num_processors", str(os.cpu_count()),
               "--job_manager", "multiprocessing"]
    rss = os.path.join(out, f"{tool}.maxrss")
    peak_sum = 0
    t = time.perf_counter()
    with open(os.path.join(out, f"{tool}.log"), "w") as log:
        # GNU time %M: the largest single process (wait4 folds in waited-for descendants)
        p = subprocess.Popen(["/usr/bin/time", "-f", "%M", "-o", rss] + cmd, stdout=log, stderr=subprocess.STDOUT)
        ps = psutil.Process(p.pid)
        while p.poll() is None:  # whole process tree, sampled every 0.5 s
            peak_sum = max(peak_sum, _tree_pss(ps))
            time.sleep(0.5)
    wall = time.perf_counter() - t
    if p.returncode:
        raise SystemExit(f"{tool} exited with {p.returncode}; see {out}/{tool}.log")
    peak = int(open(rss).read().split()[-1]) / 1024 ** 2
    inputs, models = count(sdf, "SMILES" if tool == "gypsum" else None)  # Gypsum's first record holds parameters
    return {"command": cmd, "wall_s": round(wall, 1), "peak_rss_gb_largest_process": round(peak, 2),
            "peak_pss_gb_process_tree_sampled": round(peak_sum / 1024 ** 3, 2),
            "inputs_with_models": inputs, "models": models}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("smi")
    ap.add_argument("--tools", default="vsgen-cpu,vsgen-gpu,gypsum")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    path = os.path.join(a.out, "speed.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    n = sum(1 for l in open(a.smi) if l.strip())
    for tool in a.tools.split(","):
        if tool in res and not a.force:
            print(f"{tool}: in {path}, skipped")
            continue
        r = run_tool(tool, os.path.abspath(a.smi), a.out)
        r.update(inputs=n, input_file=os.path.abspath(a.smi), environment=environment(tool),
                 finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        res[tool] = r
        json.dump(res, open(path, "w"), indent=1)
        print(f"{tool}: {r['wall_s'] / 60:.1f} min, {r['inputs_with_models']}/{n} inputs, {r['models']} models", flush=True)
    base = res.get("gypsum", {}).get("wall_s")
    for tool, r in res.items():
        print(f"{tool:10s} {r['wall_s'] / 60:7.1f} min  {base / r['wall_s'] if base else float('nan'):5.1f}x  "
              f"{r['inputs_with_models']} inputs  {r['models']} models")


if __name__ == "__main__":
    main()
