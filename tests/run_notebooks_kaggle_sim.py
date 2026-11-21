"""Execute all seven notebooks in a simulated Kaggle file layout with the CPU 'mock' profile.

    python tests/run_notebooks_kaggle_sim.py

Creates /kaggle/working and /kaggle/input, "attaches" the code dataset and the previous notebook's
output exactly like Kaggle does, and runs the notebooks one after another with nbclient.
"""
from __future__ import annotations

import glob
import random
import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parent.parent
K = Path("/kaggle")


def fill_annotations(src_dir: Path, dst_dir: Path) -> None:
    from openpyxl import load_workbook

    rng = random.Random(0)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.glob("human_eval_*_annotator*.xlsx")):
        wb = load_workbook(f)
        ws = wb["Annotate"]
        for row in ws.iter_rows(min_row=2):
            if row[0].value and not str(row[0].value).startswith("ARTICLE"):
                lab = rng.choice(["Supported", "Supported", "Partial", "Hallucinated"])
                row[5].value = lab
                if lab != "Supported":
                    row[6].value = rng.choice(["Number", "Entity", "Extrinsic (not in article)"])
                if row[3].value == 1:
                    row[7].value, row[8].value = rng.randint(2, 5), rng.randint(2, 5)
        wb.save(dst_dir / f.name)


def main() -> None:
    shutil.rmtree(K, ignore_errors=True)
    (K / "working").mkdir(parents=True)
    (K / "input").mkdir(parents=True)
    shutil.copytree(REPO, K / "input" / "factindic-code" / "factindic",
                    ignore=shutil.ignore_patterns(".git", "factindic_out", "__pycache__", "notebooks", "tests"))
    names = sorted(p.stem for p in (REPO / "notebooks").glob("*.ipynb"))
    prev = None
    for name in names:
        # fresh session: empty working dir, inputs = code + previous output
        for p in (K / "working").iterdir():
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        for p in (K / "input").iterdir():
            if p.name.startswith("out-") and p.name != f"out-{prev}":
                shutil.rmtree(p)
        if name.startswith("07"):
            fill_annotations(K / "input" / f"out-{prev}" / "factindic_out" / "06_evaluate" / "human_eval",
                             K / "input" / "filled-annotations")
        nb = nbformat.read(REPO / "notebooks" / f"{name}.ipynb", as_version=4)
        for c in nb.cells:
            if c.cell_type == "code":
                c.source = c.source.replace('PROFILE = "full"', 'PROFILE = "mock"')
        t0 = time.time()
        NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(K / "working")}}).execute()
        nbformat.write(nb, K / "working" / f"{name}.executed.ipynb")
        errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
        print(f"{name}: OK in {time.time() - t0:.0f}s, {len(errors)} errors")
        # Kaggle: the notebook's output becomes an attachable input for the next one
        out = K / "input" / f"out-{name}"
        shutil.copytree(K / "working", out)
        prev = name
    print("stages in final output:", sorted(p.name for p in (K / "input" / f"out-{prev}" / "factindic_out").iterdir()))


if __name__ == "__main__":
    sys.exit(main())
