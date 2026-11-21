"""Moving stage outputs between Kaggle notebooks.

Each notebook writes to ``/kaggle/working/factindic_out/<stage>/``. At start-up a notebook copies
every earlier stage it can find under ``/kaggle/input`` (the attached output of the previous
notebook) into its own working folder, so its output again contains *all* stages so far.
That way each notebook only needs the previous notebook attached.
"""
from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path

STAGES = ["01_data", "02_codemix", "03_seq2seq", "04_llm", "05_verify", "06_evaluate", "07_human", "07_llm_judge"]


def input_roots() -> list[Path]:
    roots = []
    extra = os.environ.get("FACTINDIC_IN")
    if extra:
        roots += [Path(p) for p in extra.split(os.pathsep) if p]
    for p in glob.glob("/kaggle/input/**/factindic_out", recursive=True):
        roots.append(Path(p))
    return roots


def bring_forward(cfg, log=print) -> list[str]:
    """Copy earlier stages from attached inputs into ``cfg.out_root``. Returns the stages found."""
    out = Path(cfg.out_root)
    out.mkdir(parents=True, exist_ok=True)
    found = []
    for root in input_roots():
        for st in STAGES:
            src = root / st
            dst = out / st
            if src.is_dir() and not dst.exists():
                shutil.copytree(src, dst)
                found.append(st)
    found = sorted(set(found) | {st for st in STAGES if (out / st).is_dir()})
    log(f"Stages available: {', '.join(found) if found else 'none'}")
    return found


def require(cfg, *stages: str) -> None:
    missing = [s for s in stages if not (Path(cfg.out_root) / s).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing outputs of stage(s) {missing}. Attach the previous notebook's output "
            "(right panel -> Add Input -> Your Work -> the previous notebook) and run again.")


def cleanup_jobs(cfg) -> None:
    """Remove worker scratch files and the resume cache once a stage has saved its results."""
    shutil.rmtree(Path(cfg.out_root) / "_jobs", ignore_errors=True)
    shutil.rmtree(Path(cfg.out_root) / "_cache", ignore_errors=True)


def stage_done(cfg, stage: str) -> bool:
    """True once a stage has finished (each stage writes its config.json last)."""
    return (Path(cfg.out_root) / stage / "config.json").exists()


def record_env(cfg, stage: str) -> dict:
    """Save which platform/GPUs a stage ran on (used to describe the setup correctly in the paper)."""
    import json
    import os
    import subprocess
    import sys

    from .llm import choose_quantization

    gpus, mem = [], []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        for line in out:
            name, m = [x.strip() for x in line.split(",")]
            gpus.append(name)
            mem.append(float(m) / 1024)
    except Exception:
        pass
    platform = "colab" if ("google.colab" in sys.modules or os.path.isdir("/content/drive")) else (
        "kaggle" if os.path.isdir("/kaggle/working") else "local")
    total = sum(mem)
    info = {"platform": platform, "gpus": gpus, "gpu_mem_gb": mem,
            "rewriter": cfg.codemix_model, "summarizer": cfg.summarizer_model,
            "rewriter_quantization": choose_quantization(cfg.codemix_model, cfg.quantize, total) if total else None,
            "summarizer_quantization": choose_quantization(cfg.summarizer_model, cfg.quantize, total) if total else None,
            "backend": cfg.backend}
    p = Path(cfg.out_root) / stage / "env.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
