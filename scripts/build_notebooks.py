"""Generate the seven Kaggle notebooks in notebooks/ (run after editing this file).

    python scripts/build_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent

SETUP = r'''# ── Setup: make the factindic code importable ─────────────────────────────────────────
# Kaggle: upload factindic-code.zip as a private dataset and attach it
#         (right-hand panel -> Add Input -> Your Work / Datasets -> factindic-code -> "+").
# Colab:  put factindic-code.zip in My Drive (and run the Google Drive cell first), or upload it to /content.
# Or:     push the repository to GitHub and paste its URL here (needs Internet).
REPO_URL = ""   # e.g. "https://github.com/<your-username>/factindic"

import glob, os, shutil, subprocess, sys, zipfile

ON_COLAB = "google.colab" in sys.modules
SRC = "/content/factindic_src" if ON_COLAB else "/kaggle/working/factindic_src"

def _is_pkg(d):
    return all(os.path.exists(os.path.join(d, f)) for f in ("__init__.py", "pipeline.py", "config.py"))

def _candidates():
    inits = glob.glob("/kaggle/input/**/__init__.py", recursive=True)
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    if ON_COLAB:  # shallow search only: Google Drive can be huge
        for base in ("/content", "/content/drive/MyDrive"):
            for depth in ("", "*/", "*/*/"):
                inits += glob.glob(f"{base}/{depth}factindic/__init__.py")
            zips += [z for z in glob.glob(f"{base}/*.zip") + glob.glob(f"{base}/*/*.zip")
                     if "factindic" in os.path.basename(z).lower()]
    return sorted(set(inits)), sorted(set(zips))

def _unzip(z):
    try:
        with zipfile.ZipFile(z) as zf:
            hits = [n for n in zf.namelist() if n.endswith("factindic/__init__.py")]
            if hits:
                shutil.rmtree(SRC, ignore_errors=True)
                zf.extractall(SRC)
                return os.path.dirname(os.path.dirname(os.path.join(SRC, sorted(hits, key=len)[-1])))
    except zipfile.BadZipFile:
        pass
    return None

def _find_code():
    inits, zips = _candidates()
    # 0) Colab: always unpack the newest zip from Drive, so replacing the zip there updates the code
    if ON_COLAB:
        for z in sorted(zips, key=os.path.getmtime, reverse=True):
            code = _unzip(z)
            if code:
                print("using", z)
                return code
    # 1) the repository folder (…/factindic/factindic/__init__.py)
    for p in inits:
        d = os.path.dirname(p)
        if _is_pkg(d) and os.path.basename(d) == "factindic":
            return os.path.dirname(d)
    # 2) only the inner package was uploaded (files directly in the dataset): copy it under the right name
    for p in inits:
        d = os.path.dirname(p)
        if _is_pkg(d):
            shutil.copytree(d, os.path.join(SRC, "factindic"), dirs_exist_ok=True)
            for req in glob.glob(os.path.join(os.path.dirname(d), "**", "requirements.txt"), recursive=True)[:1]:
                shutil.copy(req, SRC)
            return SRC
    # 3) a zip that nobody unpacked
    for z in zips:
        code = _unzip(z)
        if code:
            return code
    # 4) running locally from the repository
    for p in ("factindic", "../factindic"):
        if _is_pkg(p):
            return os.path.abspath(os.path.dirname(p) or ".")
    return None

CODE = _find_code()
if CODE is None and REPO_URL:
    subprocess.run(["git", "clone", "-q", "--depth", "1", REPO_URL, SRC], check=True)
    CODE = SRC
if CODE is None:
    where = (sorted(glob.glob("/content/*")) + sorted(glob.glob("/content/drive/MyDrive/*"))[:30]) if ON_COLAB else \
            (sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*"))[:30])
    how = ("Put factindic-code.zip in My Drive and run the Google Drive cell first (or upload the zip to /content)."
           if ON_COLAB else
           "Attach the dataset: right-hand panel -> Add Input -> Your Work (or Datasets) -> factindic-code -> '+'. "
           "If you uploaded it a minute ago, wait until Kaggle finishes processing it.")
    raise FileNotFoundError("factindic code not found.\n" + how + "\nWhat this notebook can see:\n  "
                            + ("\n  ".join(where) if where else "(nothing)"))
sys.path.insert(0, CODE)
os.environ["PYTHONPATH"] = CODE + os.pathsep + os.environ.get("PYTHONPATH", "")
req = os.path.join(CODE, "requirements.txt")
pkgs = ["-r", req] if os.path.exists(req) else ["indic-transliteration", "wordfreq", "datasketch", "bert-score",
                                                "python-docx", "feedparser", "trafilatura", "peft", "openpyxl",
                                                "bitsandbytes"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
import factindic
from factindic import Config
from factindic import pipeline as P
print("factindic", factindic.__version__, "loaded from", CODE)'''

PROFILE = r'''# ── Configuration ────────────────────────────────────────────────────────────────────
# Use the SAME profile in every notebook of the chain.
#   "full"  – the real experiment
#   "quick" – ~25 articles per language, to check the whole chain on Kaggle first
#   "mock"  – CPU dry run with stand-in models (no GPU, no downloads); never report its numbers
PROFILE = "full"
cfg = Config.make(PROFILE)
{extra}print(cfg.to_json())'''

REPORT = r'''from IPython.display import Markdown, display
display(Markdown(open(f"{{cfg.out_root}}/{stage}/{name}", encoding="utf-8").read()))'''

DONE = r'''import os
for root, dirs, files in os.walk(cfg.out_root):
    dirs[:] = [d for d in dirs if not d.startswith("_")]
    depth = root[len(cfg.out_root):].count(os.sep)
    if depth <= 1:
        print("  " * depth + os.path.basename(root) + "/", ", ".join(sorted(files))[:300])'''


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def header(num: str, title: str, purpose: str, accel: str, internet: str, attach: str, time: str) -> str:
    return f"""# FactIndic {num} – {title}

{purpose}

| Setting (right-hand panel) | Value |
|---|---|
| Accelerator | **{accel}** |
| Internet | **{internet}** |
| Inputs to attach | {attach} |
| Time (full profile) | {time} |

Run it with **Save Version → Save & Run All (Commit)**. When it finishes, its output (folder `factindic_out`) is
ready to attach to the next notebook. Each notebook's output contains every earlier stage, so you only ever attach
the previous notebook."""


NOTEBOOKS = {
    "01_data_preparation": [
        md(header("01", "Data collection and preprocessing",
                  "Loads XL-Sum Hindi and Bengali (BBC article–summary pairs), cleans the articles (BBC boilerplate, "
                  "Unicode, digits), filters by length and script, removes exact and near-duplicates (MinHash, "
                  "5-gram Jaccard ≥ 0.8) between test and training articles, and samples the test, dev and training "
                  "articles. It also collects a small set of naturally code-mixed news (RSS, best effort) and any "
                  "documents you add yourself. XL-Sum's official splits are kept (the public mT5/IndicBART "
                  "checkpoints were trained on its training split); collected documents are split 70/10/20 by article.",
                  "None (CPU)", "On", "the `factindic-code` dataset; optionally a dataset with your own code-mixed "
                  "documents (a CSV/JSONL whose file name contains `natural`, columns `lang`, `text`)", "10–25 min")),
        code(SETUP),
        code(PROFILE.format(extra="# cfg.scrape = False   # turn off RSS scraping\n")),
        code("docs = P.run_stage01(cfg)"),
        code(REPORT.format(stage="01_data", name="data_report.md")),
        md("A few selected articles and their reference summaries:"),
        code('docs[docs.role == "test"].groupby("lang").head(2)[["doc_id", "lang", "n_words", "text", "summary_ref"]]'),
        code(DONE),
    ],
    "02_codemix_generation": [
        md(header("02", "Code-mixed input variants",
                  "An LLM rewrites every test (and LoRA training) article sentence by sentence into code-mixed text "
                  "(native-script grammar, English content words in Latin letters, e.g. *भारत में pollution का level "
                  "लगातार increase हो रहा है*), then transliterates that into fully romanised Hinglish/Banglish. "
                  "Each sentence is checked (numbers unchanged, length, script) and retried once; articles with "
                  "under 80% valid sentences are excluded from the paired comparison. The log prints the batch size and "
                  "an estimated time left every few minutes. "
                  "Also writes `codemix_quality_check.xlsx` for a quick human check of meaning and naturalness.",
                  "GPU T4 ×2", "On", "output of notebook 01", "about 1–2 h; the log prints an estimate")),
        code(SETUP),
        code(PROFILE.format(extra="# cfg.codemix_model = \"Qwen/Qwen3-8B\"   # a different, larger rewriter: about 4x slower\n")),
        code("var = P.run_stage02(cfg)"),
        code(REPORT.format(stage="02_codemix", name="codemix_report.md")),
        md("Three versions of the same sentence:"),
        code('''t = var[var.role == "test"]
for doc_id in t.doc_id.unique()[:3]:
    g = t[t.doc_id == doc_id].set_index("variant")
    for v in ("mono", "cm", "roman"):
        if v in g.index:
            print(f"{v:6s}", g.at[v, "sents"][0])
    print()'''),
        code(DONE),
    ],
    "03_seq2seq_baselines": [
        md(header("03", "Fine-tuned summarisers (mT5, IndicBART) and LoRA",
                  "Summarises every test input (mono, cm, roman, natural) with `csebuetnlp/mT5_multilingual_XLSum` "
                  "and `ai4bharat/IndicBART-XLSum`, then trains a LoRA adapter on the mT5 checkpoint with the "
                  "training articles in all three variants and evaluates it the same way (`mt5-lora-cm`). "
                  "If IndicBART fails to load with your library versions the notebook continues without it.",
                  "GPU T4 (one is enough)", "On", "output of notebook 02", "30–45 min")),
        code(SETUP),
        code(PROFILE.format(extra="# cfg.seq2seq_models = (\"mt5\",)   # drop IndicBART\n# cfg.lora = False\n")),
        code("s2s = P.run_stage03(cfg)"),
        code('s2s.groupby(["system", "lang", "variant"]).agg(n=("summary", "count"), time_s=("time_s", "mean"))'),
        code('s2s.groupby("system").head(2)[["system", "lang", "variant", "summary"]]'),
        code(DONE),
    ],
    "04_llm_summarization": [
        md(header("04", "LLM summarisation: Track A, Track B, Track BV",
                  "**Track A** – standard prompt. **Track B** – factuality-aware prompt: entity locking, a list of "
                  "the article's numbers and dates that the summary may use (extracted by code), and chain-of-fact "
                  "(FACTS → ENTITIES → SUMMARY). **Track BV** – Track B plus verify-and-revise: a rule-based "
                  "number/date check and an LLM self-check flag unsupported sentences, and the model rewrites the "
                  "summary once. Same model, length budget and greedy decoding for every track. The notebook then "
                  "generates questions from every summary (all systems) for the QA-based check in notebook 05. "
                  "With a 4B model on two T4s, one worker runs on each GPU.",
                  "GPU T4 ×2", "On", "output of notebook 03", "about 4–6 h; the log prints an estimate")),
        code(SETUP),
        code(PROFILE.format(extra='''# cfg.summarizer_model = "Qwen/Qwen3-8B"   # bigger model: slower (one process over both GPUs)
# Optional: an OpenAI-compatible API instead of a local model (key stored as a Kaggle secret OPENAI_API_KEY)
# from kaggle_secrets import UserSecretsClient
# os.environ["OPENAI_API_KEY"] = UserSecretsClient().get_secret("OPENAI_API_KEY")
# cfg.backend, cfg.openai_base_url, cfg.openai_model = "openai", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"
''')),
        md("Optional speed-up: vLLM is several times faster than plain transformers, but installing it replaces "
           "Kaggle's PyTorch and can break other packages. Only switch it on if you have time to debug; the default "
           "(transformers) works out of the box."),
        code('''USE_VLLM = False
if USE_VLLM:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)
    cfg.backend = "vllm"'''),
        code("llm = P.run_stage04(cfg)"),
        code(REPORT.format(stage="04_llm", name="llm_report.md")),
        md("Same article, three inputs, three tracks:"),
        code('''one = llm[llm.role == "test"].doc_id.iloc[0]
for r in llm[llm.doc_id == one].sort_values(["variant", "track"]).itertuples():
    print(f"[{r.variant:5s} {r.track:2s}] {r.summary}")'''),
        code(DONE),
    ],
    "05_factuality_verification": [
        md(header("05", "Factuality verification layer",
                  "Checks every summary (all systems and tracks) against the **original monolingual article** "
                  "(primary – the facts are identical across inputs, so this measures the summariser and not how "
                  "well the checker reads code-mixed text) and against the model's actual input (secondary). "
                  "(1) multilingual NLI entailment per summary sentence (mDeBERTa-v3, SummaC-style); "
                  "(2) QA-based consistency (questions from the summary answered from the article by XLM-R); "
                  "(3) entity, numeral and date alignment (IndicNER + cross-script matching) giving precision "
                  "against the source and precision/recall/F1 against the reference; plus an error type per "
                  "unsupported sentence.",
                  "GPU T4", "On", "output of notebook 04", "20–40 min")),
        code(SETUP),
        code(PROFILE.format(extra="")),
        code("ver = P.run_stage05(cfg)"),
        code(REPORT.format(stage="05_verify", name="verify_report.md")),
        code(DONE),
    ],
    "06_evaluation_analysis": [
        md(header("06", "Evaluation, hypothesis tests, code-mixing analysis, figures",
                  "Computes ROUGE-1/2/L (P/R/F1, Unicode tokenizer), METEOR, BERTScore (P/R/F1), LaBSE similarity, "
                  "average summary length, compression ratio and inference time; merges the factuality scores; "
                  "tests **H1** (code-mixed vs monolingual input, paired Wilcoxon/McNemar with bootstrap CIs, Holm "
                  "correction) and **H2** (Track B/BV vs A, and whether they shrink the gap: difference-in-differences); "
                  "analyses factuality by Code-Mixing Index, tokenizer fertility, script of the output and error "
                  "types; draws the figures; and writes the blind human-evaluation sheets.",
                  "GPU T4", "On", "output of notebook 05", "15–30 min")),
        code(SETUP),
        code(PROFILE.format(extra="")),
        md("Optional: the official XL-Sum ROUGE scorer (language-specific stemming) so your ROUGE is comparable "
           "with published XL-Sum numbers. If it does not install, the notebook reports the Unicode ROUGE only."),
        code('''INSTALL_XLSUM_ROUGE = True
if INSTALL_XLSUM_ROUGE and PROFILE != "mock":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyonmttok",
                    "git+https://github.com/csebuetnlp/xl-sum.git#subdirectory=multilingual_rouge_scoring"], check=False)'''),
        code("res = P.run_stage06(cfg)"),
        code(REPORT.format(stage="06_evaluate", name="results.md")),
        code('''from IPython.display import Image, display
import glob
for f in sorted(glob.glob(f"{cfg.out_root}/06_evaluate/figures/*.png")):
    print(os.path.basename(f)); display(Image(f))'''),
        md("**Paper draft.** The cell below writes the IEEE-style paper (Word) with your numbers, tables and figures "
           "filled in: `06_evaluate/FactIndic_paper_draft.docx`. Yellow `[Write: ...]` notes mark the sentences that "
           "interpret the results; write those yourselves. Notebook 07 rebuilds it with the human evaluation."),
        code('''from factindic.paper import build_paper
print(build_paper(cfg.out_root, f"{cfg.out_root}/06_evaluate/FactIndic_paper_draft.docx"))'''),
        md("**Next:** download `06_evaluate/human_eval/` from this notebook's Output tab. Send each "
           "`human_eval_<lang>_annotator<k>.xlsx` to one annotator, and keep `human_eval_KEY_do_not_share.csv` to "
           "yourself. When the files come back filled in, upload them as a Kaggle dataset and run notebook 07."),
        code(DONE),
    ],
    "07_human_evaluation": [
        md(header("07", "Human evaluation analysis",
                  "Reads the filled-in annotation sheets and reports: inter-annotator agreement (Cohen's kappa, target "
                  "> 0.6), how well each automatic checker detects human-labelled hallucinations (precision, recall, "
                  "F1), Spearman correlation between automatic scores and human judgements, human hallucination rates "
                  "per track and input type (a human check of H1 and H2), fluency and coverage ratings, and error types.",
                  "None (CPU)", "On", "output of notebook 06 **and** a dataset with the filled-in "
                  "`human_eval_*_annotator*.xlsx` files", "2 min")),
        code(SETUP),
        code(PROFILE.format(extra="")),
        code("out = P.run_stage07(cfg)"),
        md("**LLM-judge labels.** If the attached sheets were filled in by an LLM judge (the folder then contains "
           "`judge.json`), this notebook writes to `07_llm_judge`, leaves out inter-annotator agreement, and the "
           "paper describes the result as LLM-judge evaluation and never as human evaluation."),
        md("The final paper draft, now including this evaluation (download it from the Output tab):"),
        code('''from factindic.paper import build_paper
print(build_paper(cfg.out_root, f"{cfg.out_root}/FactIndic_paper_draft.docx"))'''),
        code(DONE),
    ],
}


def main():
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        nbf.write(nb, out / f"{name}.ipynb")
        print("wrote", out / f"{name}.ipynb")


if __name__ == "__main__":
    main()
