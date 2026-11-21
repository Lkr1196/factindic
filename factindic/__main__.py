"""Command-line entry point for running stages outside notebooks.

    python -m factindic 01 --profile quick
    python -m factindic all --profile mock --out ./factindic_out
    python -m factindic 07 --profile full --annotations filled/*.xlsx
"""
from __future__ import annotations

import argparse
import glob
import os


def main() -> None:
    ap = argparse.ArgumentParser(prog="factindic")
    ap.add_argument("stage", help="01..07 or 'all' (01-06)")
    ap.add_argument("--profile", default="full", choices=["full", "quick", "mock"])
    ap.add_argument("--out", default=None, help="output folder (default: ./factindic_out or /kaggle/working/...)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. n_test=50")
    ap.add_argument("--annotations", nargs="*", default=None, help="filled annotation .xlsx files (stage 07)")
    args = ap.parse_args()
    if args.out:
        os.environ["FACTINDIC_OUT"] = os.path.abspath(args.out)
    from . import pipeline as P
    from .config import Config

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        default = getattr(Config(), k)
        if isinstance(default, bool):
            v = v.lower() in ("1", "true", "yes")
        elif isinstance(default, int):
            v = int(v)
        elif isinstance(default, float):
            v = float(v)
        elif isinstance(default, tuple):
            v = tuple(x for x in v.split(",") if x)
        overrides[k] = v
    cfg = Config.make(args.profile, **overrides)
    stages = ["01", "02", "03", "04", "05", "06"] if args.stage == "all" else [args.stage.zfill(2)]
    for st in stages:
        print(f"\n===== stage {st} =====")
        if st == "07":
            files = [f for pat in (args.annotations or []) for f in glob.glob(pat)]
            P.run_stage07(cfg, files or None)
        else:
            getattr(P, f"run_stage{st}")(cfg)


if __name__ == "__main__":
    main()
