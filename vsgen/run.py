"""Streaming scheduler: CPU process pool + optional GPU worker process.

Variants wait in one cost-sorted pool. CPU workers take the most expensive
first (longest-processing-time scheduling: the slow tail starts early instead
of idling cores at the end); the GPU worker takes batches of the cheapest,
which pack well. Inputs are read in windows so memory stays bounded, and
models are written in input order.
"""

import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field

from rdkit import Chem, RDLogger

from vsgen import chem, conf, pka_unipka

RDLogger.DisableLog("rdApp.*")


@dataclass
class RunConfig:
    chemistry: chem.Options = field(default_factory=chem.Options)
    three_d: conf.Conf3D = field(default_factory=conf.Conf3D)
    procs: int = 0  # 0 = all cores (minus GPU feeder cores when the GPU is used)
    gpu: str = "auto"  # auto | on | off
    gpu_batch_variants: int = 256
    window: int = 4000  # variants held in memory at once
    protomers_from: str = ""  # vsgen-protomers cache (Uni-pKa microstates) instead of the pKa model
    protomers_fallback: bool = False  # use --pka-model for inputs the cache does not cover
    two_d_chunk: int = 64


def read_inputs(path):
    """Yield (index, name, smiles, props) from .smi/.smiles/.txt or .sdf."""
    if path.lower().endswith(".sdf"):
        for i, m in enumerate(Chem.SDMolSupplier(path)):
            if m is None:
                yield i, f"record{i}", "", {}
                continue
            name = m.GetProp("_Name") if m.HasProp("_Name") and m.GetProp("_Name") else f"mol{i}"
            yield i, name, Chem.MolToSmiles(m), {k: str(v) for k, v in m.GetPropsAsDict().items()}
        return
    with open(path) as f:
        i = 0
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            yield i, parts[1] if len(parts) > 1 else f"mol{i}", parts[0], {}
            i += 1


def _two_d(batch, opts, cache, require_cache):
    out = []
    for i, name, smi, _ in batch:
        states = cache.get(chem.cache_key(smi))
        if require_cache and states is None:  # missing entry, or one the cache marks as failed
            out.append((i, ([], "no usable entry in the protomer cache; rerun vsgen-protomers "
                                "for this input or pass --protomers-fallback")))
            continue
        out.append((i, chem.prepare_2d(i, name, smi, opts, states)))
    return out


def _cpu_task(variant, cfg):
    models, err = conf.build_cpu(variant.smiles, variant.seed, cfg)
    return variant, [(m, e, k, "cpu") for m, e, k in models], err


def _gpu_task(variants, cfg):
    res = conf.build_gpu_batch([(v.smiles, v.seed) for v in variants], cfg)
    return variants, res


def _gpu_ok():
    try:
        import nvmolkit.embedMolecules  # noqa: F401
        import subprocess

        return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
    except Exception:
        return False


def _cost(v):
    return Chem.AddHs(Chem.MolFromSmiles(v.smiles)).GetNumAtoms() ** 2


class _Writer:
    """Writes each input's models once it and every earlier input are done."""

    def __init__(self, out_path, failed_path):
        self.sdf = Chem.SDWriter(out_path)
        self.failed = open(failed_path, "w")
        self.failed.write("index\tname\tinput_smiles\tvariant_smiles\treason\n")
        self.pending, self.done, self.props = {}, {}, {}
        self.next, self.models, self.inputs_ok, self.truncated = 0, 0, 0, 0

    def expect(self, index, n, props):
        self.pending[index] = n
        self.done[index] = []
        self.props[index] = props

    def add(self, index, item):
        self.done[index].append(item)
        self.pending[index] -= 1

    def fail(self, index, name, smi, variant_smi, reason):
        self.failed.write(f"{index}\t{name}\t{smi}\t{variant_smi}\t{reason}\n")

    def flush(self):
        while self.next in self.pending and self.pending[self.next] == 0:
            i = self.next
            items = sorted(self.done.pop(i), key=lambda it: (it[0].lineage, it[0].smiles))
            for v, models in items:
                for n, (m, energy, kind, backend) in enumerate(models):
                    for k, val in self.props.get(i, {}).items():
                        m.SetProp(k, val)
                    m.SetProp("_Name", v.name)
                    m.SetProp("SMILES", v.smiles)
                    m.SetProp("input_smiles", v.input_smiles)
                    m.SetProp("lineage", v.lineage)
                    m.SetProp("model", f"{kind}{n}" if n else kind)
                    m.SetProp("backend", backend)
                    m.SetProp("seed", str(v.seed))
                    if v.stereo_undefined:
                        # so nobody reads an enumerated guess as an asserted structure
                        m.SetIntProp("stereo_undefined", v.stereo_undefined)
                        m.SetProp("stereo_coverage", "complete" if v.stereo_emitted >= v.stereo_total
                                  else f"truncated ({v.stereo_emitted} of {v.stereo_total} stereoisomers)")
                    if energy is not None:
                        m.SetDoubleProp("energy", round(energy, 3))  # kcal/mol, in the force_field property's force field
                    self.sdf.write(m)
                    self.models += 1
            self.inputs_ok += any(models for _, models in items)
            self.truncated += any(v.stereo_emitted < v.stereo_total for v, models in items if models)
            del self.pending[i]
            self.props.pop(i, None)
            self.next += 1

    def close(self):
        self.flush()
        self.sdf.close()
        self.failed.close()


def run(input_path, out_path, cfg=RunConfig(), log=sys.stderr):
    """Prepare every input; returns a summary dict."""
    t0 = time.perf_counter()
    use_gpu = cfg.gpu == "on" or (cfg.gpu == "auto" and _gpu_ok())
    if cfg.gpu == "on" and not _gpu_ok():
        raise RuntimeError("--gpu on, but nvMolKit or an NVIDIA GPU is not available")
    n_cpu = cfg.procs or max(1, mp.cpu_count() - (4 if use_gpu else 0))
    protomer_cache = pka_unipka.load_cache(cfg.protomers_from) if cfg.protomers_from else {}
    if cfg.protomers_from:
        failed = sum(v is None for v in protomer_cache.values())
        print(f"protomer cache: {len(protomer_cache) - failed} inputs from {cfg.protomers_from}"
              + (f", {failed} recorded as failed" if failed else "")
              + (" (uncached inputs fall back to --pka-model)" if cfg.protomers_fallback else ""), file=log)
    failed_path = out_path.rsplit(".", 1)[0] + "_failed.tsv"
    writer = _Writer(out_path, failed_path)
    inputs = read_inputs(input_path)
    exhausted = False
    queue = []  # variants sorted by cost, most expensive first
    names = {}  # index -> (name, smiles) for failure reports
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")  # one BLAS thread per worker; 16 workers x 16 threads thrash
    ctx = mp.get_context("spawn")  # nvMolKit/CUDA and RDKit are not fork-safe together
    cpu = ProcessPoolExecutor(n_cpu, mp_context=ctx)
    gpu = ProcessPoolExecutor(1, mp_context=ctx) if use_gpu else None
    cpu_busy, gpu_busy, gpu_take, two_d = {}, None, [], set()
    n_inputs = n_variants = 0
    last_log = t0

    def cpu_submit(v):
        cpu_busy[cpu.submit(_cpu_task, v, cfg.three_d)] = v

    try:
        while True:
            # Keep the 2D stage ahead of the 3D stage, within the memory window.
            while not exhausted and len(two_d) < 2 and len(queue) + len(cpu_busy) < cfg.window:
                batch = []
                for _ in range(cfg.two_d_chunk):
                    item = next(inputs, None)
                    if item is None:
                        exhausted = True
                        break
                    batch.append(item)
                    names[item[0]] = (item[1], item[2])
                    writer.expect(item[0], 1, item[3])  # placeholder until 2D reports
                    n_inputs += 1
                if batch:
                    chunk_cache = {}  # only this chunk's entries travel to the worker
                    for _, _, smi, _ in batch:
                        key = chem.cache_key(smi)
                        if key in protomer_cache:
                            chunk_cache[key] = protomer_cache[key]
                    two_d.add(cpu.submit(_two_d, batch, cfg.chemistry, chunk_cache,
                                         bool(cfg.protomers_from) and not cfg.protomers_fallback))
            # GPU takes the cheapest variants in batches.
            if gpu is not None and gpu_busy is None and queue and (len(queue) >= cfg.gpu_batch_variants or exhausted):
                take = []
                while queue and len(take) < cfg.gpu_batch_variants:
                    v = queue.pop()
                    if conf.gpu_supported(v.smiles, cfg.three_d):
                        take.append(v)
                    else:
                        cpu_submit(v)  # metals / missing UFF params: CPU only
                if take:
                    gpu_busy, gpu_take = gpu.submit(_gpu_task, take, cfg.three_d), take
            # CPU takes the most expensive; few in flight so the order holds.
            while queue and len(cpu_busy) < 2 * n_cpu:
                cpu_submit(queue.pop(0))
            pending = set(cpu_busy) | two_d | ({gpu_busy} if gpu_busy else set())
            if not pending:
                if exhausted and not queue:
                    break
                continue
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            new = []
            for fut in done:
                if fut in two_d:
                    two_d.discard(fut)
                    for i, (variants, err) in fut.result():
                        writer.pending[i] = len(variants)
                        if err:
                            writer.fail(i, *names[i], "", err)
                        new.extend(variants)
                        n_variants += len(variants)
                elif fut is gpu_busy:
                    gpu_busy = None
                    try:
                        variants, results = fut.result()
                    except Exception as exc:  # whole batch failed: CPU redoes it
                        print(f"GPU batch failed ({exc}); falling back to CPU", file=log)
                        variants, results = gpu_take, [None] * len(gpu_take)
                    for v, res in zip(variants, results):
                        if res is None:
                            cpu_submit(v)
                        else:
                            writer.add(v.input_index, (v, [(m, e, k, "gpu") for m, e, k in res[0]]))
                else:
                    v = cpu_busy.pop(fut)
                    try:
                        v, models, err = fut.result()
                    except Exception as exc:
                        models, err = [], f"worker error: {exc}"
                    if err:
                        writer.fail(v.input_index, v.name, v.input_smiles, v.smiles, err)
                    writer.add(v.input_index, (v, models))
            if new:
                queue.extend(new)
                queue.sort(key=_cost, reverse=True)
            writer.flush()
            for i in [i for i in names if i < writer.next]:
                del names[i]
            now = time.perf_counter()
            if now - last_log > 10:
                print(f"  {writer.next}/{n_inputs}{'' if exhausted else '+'} inputs written, "
                      f"{writer.models} models, {writer.next / (now - t0):.1f} inputs/s", file=log, flush=True)
                last_log = now
    finally:
        cpu.shutdown(cancel_futures=True)
        if gpu is not None:
            gpu.shutdown(cancel_futures=True)
        writer.close()
    wall = time.perf_counter() - t0
    summary = {"inputs": n_inputs, "inputs_with_models": writer.inputs_ok, "variants": n_variants,
               "inputs_with_truncated_stereo": writer.truncated,
               "models": writer.models, "seconds": round(wall, 1), "inputs_per_s": round(n_inputs / wall, 2),
               "cpu_workers": n_cpu, "gpu": use_gpu, "failed_log": failed_path}
    print(f"done: {summary}", file=log)
    return summary
