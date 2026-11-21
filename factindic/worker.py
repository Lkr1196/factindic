"""Generation worker: ``python -m factindic.worker --job job.json --shard k --nshards n``.

Started by ``llm.run_generation`` (one process per GPU). Processes every n-th item, writes
results to ``out{k}.jsonl`` as it goes and prints progress lines to its log.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    job = json.load(open(args.job, encoding="utf-8"))
    from factindic.llm import load_backend

    items = []
    with open(job["items_path"], encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % args.nshards == args.shard:
                items.append(json.loads(line))
    print(f"Loading {job['model_id']} for {len(items)} prompts", flush=True)
    t0 = time.time()
    backend = load_backend(job["backend"], job["model_id"], job["cfg"])
    print(f"Loaded in {time.time() - t0:.0f}s", flush=True)
    out_path = f"{job['out_dir']}/out{args.shard}.jsonl"
    from factindic.llm import progress_line

    big = 512 if type(backend).__name__ == "VLLMBackend" else 96
    t1, s, n_tok, sizes = time.time(), 0, 0, []
    with open(out_path, "w", encoding="utf-8") as out:
        while s < len(items):
            step = 24 if s == 0 else big  # a small first chunk so the first ETA appears quickly
            part = items[s:s + step]
            res = backend.generate(part, log=lambda m: print(m, flush=True))
            for r in res:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            s += len(part)
            n_tok += sum(r.get("n_out", 0) for r in res)
            sizes += [r.get("batch_size", 1) for r in res]
            print(progress_line(f"worker {args.shard}", s, len(items), time.time() - t1, n_tok, sizes), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
