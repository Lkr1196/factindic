# FactIndic: factuality-aware summarisation for Hindi, Bengali and code-mixed input

CSET346 (Natural Language Processing) course project, Team FactForge: Aryan Singh, Lakshya Kumar, Udayveer Singh, Kushagra Agarwal and Udit Malik, Bennett University. Guided by Dr. Geetanjali Singh.

The paper and the presentation are in [`report/`](report/): [`FactIndic_paper.pdf`](report/FactIndic_paper.pdf) and [`FactIndic_presentation.pptx`](report/FactIndic_presentation.pptx). The notebooks as they ran on Kaggle, with their full output, are in [`notebooks/runs/`](notebooks/runs/).

**Research question:** can we generate concise, fluent summaries of Indic and code-mixed text that keep the facts and meaning of the original?

* **H1:** code-mixed input makes summaries measurably less factually consistent than monolingual input.
* **H2:** factuality-aware prompting reduces hallucination without hurting fluency, and closes some of the H1 gap.

The code runs on free Kaggle notebooks (2 × T4 GPU) in seven steps, and every number in the report comes from your own run.

---

## 1. The design in one picture

```
XL-Sum hi/bn test article (mono) ──LLM rewrite──► cm (code-mixed)  ──LLM transliterate──► roman (Hinglish/Banglish)
        │                                   same facts, validated sentence by sentence                │
        └───────────────┬──────────────────────────────┬───────────────────────────────────────────┘
                        ▼                              ▼
     mT5-XLSum · IndicBART-XLSum · mT5+LoRA(CM)   LLM: Track A (standard) · Track B (factuality-aware) · Track BV (B + verify & revise)
                        └──────────────┬───────────────┘
                                       ▼
        Verification against the ORIGINAL article: NLI entailment · QA consistency · entity/number/date alignment
                                       ▼
        ROUGE · METEOR · BERTScore · LaBSE · length · compression · time → paired tests (H1, H2) → figures → human evaluation
```

Each test article exists in three versions with **identical facts**:

| Input | Example |
|---|---|
| `mono` | भारत में प्रदूषण का स्तर लगातार बढ़ रहा है। |
| `cm` | भारत में pollution का level लगातार increase हो रहा है। |
| `roman` | Bharat mein pollution ka level lagatar increase ho raha hai. |

Going from `mono` to `cm` isolates the effect of **lexical mixing**. Going from `cm` to `roman` isolates the effect of **script**. Because the three versions are paired, every comparison is a paired test on the same articles.

### Design decisions to be able to defend in the viva

1. **Paired, controlled variants.** Natural code-mixed data has no parallel monolingual version, so it cannot show *how much* code-mixing costs. We rewrite real XL-Sum articles. Every rewritten sentence must keep all its numbers, a plausible length and the right script. It is retried once if it fails. Articles with less than 80% valid sentences are dropped from the paired comparison. A human check sheet (`codemix_quality_check.xlsx`) measures meaning preservation and naturalness.
2. **Factuality is checked against the original monolingual article.** The facts are the same in all three inputs, so this measures the *summariser* and not how well the checker reads code-mixed text. The check against the model's actual input is reported too, as a secondary score.
3. **The code-mixed inputs are written by an LLM** (by default the same Qwen3-4B-Instruct-2507 that summarises, which keeps the run within free GPU time). The model may find text in its own style easier than human code-mixing, so the measured gap is a lower bound; set `cfg.codemix_model = "Qwen/Qwen3-8B"` in notebook 02 for a different, larger rewriter (about 4x slower).
4. **XL-Sum's official splits are kept.** The public mT5 and IndicBART checkpoints were fine-tuned on XL-Sum's training split, so re-splitting XL-Sum 70/10/20 would leak their training articles into the test set. The deck's 70/10/20 article-level split is applied to the documents we collect ourselves. Test articles that are near-duplicates (MinHash, 5-gram Jaccard ≥ 0.8) of any training or validation article are removed.
5. **Same conditions for every track:** same model, same length budget (1–2 sentences, ≤ 40 words), and greedy decoding. Repetition penalty is off, because it would also penalise copying names and numbers from the article.
6. **Indic-aware metrics.** The usual `rouge_score` tokenizer deletes every non-ASCII character. Python's `\w` splits Hindi words at the vowel signs. Our tokenizer splits only on whitespace and punctuation, including the danda (।). Numbers are compared by value across digit systems and Indian scale words (5 लाख = 500000). Names are compared across scripts with a loose consonant skeleton, so दिल्ली = Delhi and কলকাতায় = Kolkata.

## 2. How the deck maps to the code

| Deck / problem statement | Where |
|---|---|
| Scrape → normalise → script-mix detect → deduplicate → split | `factindic/data.py`, `scrape.py`, notebook 01 |
| Hindi + Bengali (XL-Sum), code-mixed register | `config.py` (`LANGS`), `variants.py`, notebook 02 |
| Summariser: mT5 / IndicBART / LLM; LoRA on mT5-base | `seq2seq.py` (notebook 03), `summarize.py` (notebook 04) |
| Track A standard prompting | `prompts.summarize_A` |
| Track B: entity locking, number & date constraints, chain-of-fact | `prompts.summarize_B` (FACTS → ENTITIES → SUMMARY, plus a code-extracted number/date list) |
| Factuality verification layer | `verify.py` (notebook 05) + Track BV verify-and-revise in `summarize.verify_and_revise` |
| ROUGE-1/2/L, BERTScore, METEOR | `metrics.py` |
| Multilingual NLI entailment | `verify.NLIScorer` (mDeBERTa-v3, trained on Hindi and Bengali NLI among 27 languages) |
| QA-based checking (QAFactEval) | `verify.QAScorer`: a multilingual QAGS/QAFactEval-style check (QAFactEval itself is English-only) |
| Entity and numeral alignment F1 | `verify.alignment` (IndicNER + numbers + dates; precision vs source, P/R/F1 vs reference) |
| Compression ratio, average length, inference time | `metrics.length_stats`, `time_s` from the generators |
| Precision / Recall / F1 | `tables/precision_recall_f1.csv` (ROUGE, BERTScore, entities, numerals) and `07_human/detection.csv` (each checker vs. human labels) |
| Code-mixed language analysis | Code-Mixing Index (`codemix.py`), factuality vs CMI, tokenizer fertility, output script, error types (`analysis.py`) |
| 2–3 bilingual annotators, Supported/Partial/Hallucinated, double-labelling, Cohen's κ > 0.6 | `annotate.py`, sheets from notebook 06, analysis in notebook 07 |

## 3. One-time setup on Kaggle

1. Create a Kaggle account and **verify your phone number** (Settings → Phone verification). Notebooks cannot use the internet without it.
2. Upload this repository as a private dataset named **`factindic-code`**: Kaggle → Datasets → New Dataset → drag in the zip of this folder. Every notebook finds the code there. Alternatively, push the repository to GitHub and paste its URL into `REPO_URL` in the first cell of each notebook.
3. For each file in `notebooks/`: Kaggle → Create → New Notebook → File → Import Notebook. Keep the names `01_…` to `07_…`.
4. In each notebook's right-hand panel, set **Accelerator** and **Internet** as in the table below, and attach `factindic-code` (Add Input → Your Datasets).

## 4. Running the chain

Run the notebooks in order. Run each one with **Save Version → Save & Run All (Commit)**. It runs in the background, so you can close the tab. Then attach that notebook's output to the next notebook (Add Input → Your Work → the previous notebook). Each notebook's output contains every earlier stage, so you only ever attach the **previous** notebook.

| # | Notebook | Accelerator | Internet | Attach | Time (measured, full) |
|---|---|---|---|---|---|
| 01 | `01_data_preparation` | None (CPU) | On | code | 32 min |
| 02 | `02_codemix_generation` | GPU T4 ×2 | On | code + 01 | 3 h 37 min |
| 03 | `03_seq2seq_baselines` | GPU T4 | On | code + 02 | 33 min |
| 04 | `04_llm_summarization` | GPU T4 ×2 | On | code + 03 | 8 h 2 min |
| 05 | `05_factuality_verification` | GPU T4 | On | code + 04 | 35 min |
| 06 | `06_evaluation_analysis` | GPU T4 | On | code + 05 | 6 min |
| 07 | `07_human_evaluation` | None (CPU) | On | code + 06 + filled sheets | 2 min |

The times were measured on the full run with the default sizes (200 test and 60 LoRA training articles per language): 13.5 h in all, about 25 T4 GPU hours. Notebook 04 alone needs 8 h, so start it when you have that much weekly GPU quota left. The LLM notebooks (02 and 04) print a progress line every few minutes with the average batch size and an estimate of the time left. Check it after the first 15 minutes: the average batch should be well above 1. **Always use Save & Run All (Commit), never Quick Save**: Quick Save keeps no output, so the next notebook finds nothing to attach. After notebook 04, read `04_llm/llm_report.md`: the `cut_pct` and `empty_pct` columns must be near zero, otherwise summaries were cut off at the token limit (raise `cfg.max_new_tokens_A` / `cfg.max_new_tokens_B`). Kaggle gives about 30 GPU hours per week.

**Do a dry run first.** Set `PROFILE = "quick"` in every notebook. This uses about 25 articles per language and runs the real models, so the whole chain finishes in about an hour. Once it works, run everything again with `PROFILE = "full"`. `PROFILE = "mock"` runs the whole chain on CPU in about a minute with stand-in models. It tests the plumbing only, so never report its numbers.

### Changing models or sizes

All settings live in `factindic/config.py`. Override them in a notebook's configuration cell:

```python
cfg = Config.make("full")
cfg.summarizer_model = "Qwen/Qwen3-8B"   # bigger summariser (slower: one process over both GPUs)
cfg.n_test = 300                          # more test articles per language
cfg.seq2seq_models = ("mt5",)             # skip IndicBART
```

Keep `PROFILE` and any size overrides the same in every notebook. Notebook 04 also has an optional vLLM switch, which is faster but can break Kaggle's PyTorch. It also has an OpenAI-compatible API option, for example Groq. If you use the API option, describe it in the report as a different, external model.

### Adding your own naturally code-mixed documents

The deck planned 500–1,000 scraped news and social posts. Notebook 01 collects recent Hindi and Bengali news from public RSS feeds, on a best-effort basis. Social-media APIs are closed, so for posts you collect yourselves, make a CSV with columns `lang` (`hi`/`bn`) and `text`, and optionally `url`. Put it in a Kaggle dataset whose file name contains `natural`, and attach it to notebook 01. These documents have no reference summaries, so they are evaluated reference-free, on factuality only, against their own text. They appear in `tables/natural_results.csv`.

## 5. Human evaluation (slide 9)

1. Notebook 06 writes `06_evaluate/human_eval/human_eval_<lang>_annotator<k>.xlsx`. With the default settings this is 30 articles per language, each with 6 blind, shuffled summaries: Tracks A and B on mono, cm and roman inputs. 25% of the articles go to two annotators. The file `human_eval_KEY_do_not_share.csv` says which summary is which. Keep it to yourself.
2. Annotators label every summary sentence against the article as **Supported / Partial / Hallucinated**. For the last two they also give an error type. They rate each summary's fluency and coverage from 1 to 5. The instructions are inside each file, and the dropdowns are already set up.
3. Optional: one person fills `02_codemix/codemix_quality_check.xlsx`, which checks 20 sentences per language for meaning kept and naturalness.
4. Upload the filled files as a Kaggle dataset, attach it together with notebook 06's output, and run notebook 07. It reports:
   * Cohen's κ (target > 0.6)
   * precision, recall and F1 of each automatic checker against the human labels
   * Spearman correlations between automatic scores and human judgements
   * human hallucination rates per track and input, which checks H1 and H2 by hand
   * fluency and coverage ratings, and error types

### LLM-judge evaluation (when no annotator is available)

If nobody can annotate a language, the sheets can be labelled by an LLM judge that is **not** the summariser, and the result must then be reported as LLM-judge evaluation and never as human evaluation. `factindic.annotate.apply_judge_labels(sheet_dir, labels, out_dir, judge_name)` writes one label per summary sentence into copies of the sheets and adds `judge.json`. When notebook 07 finds `judge.json` next to the sheets it writes to `07_llm_judge`, leaves out Cohen's κ (each sentence has one label), and the paper builder switches every heading, caption and sentence to "LLM-judge", and adds the missing human annotation to the limitations.

## 6. Where to find things for the IEEE paper

| Paper section | Use |
|---|---|
| Dataset | `01_data/data_report.md`, `data_stats.csv` (filter counts), `02_codemix/codemix_report.md` (validity, CMI, switch rate per variant), `07_human/codemix_quality.csv` |
| Main results | `06_evaluate/results.md`, `tables/content_meaning.tex`, `factuality.tex`, `efficiency.tex`, `precision_recall_f1.csv` |
| H1 | `tables/h1_codemix_gap.csv`, `figures/codemix_gap_nli.png`, `codemix_gap_halluc.png`, `nli_support_by_input.png` |
| H2 | `tables/h2_tracks.csv` (B−A, BV−A, BV−B; quality columns show "without collapsing fluency"), `h2_gap_reduction_did.csv`, `hallucination_rate_by_input.png` |
| Code-mixing analysis | `tables/cmi_bins.csv` + `figures/nli_by_cmi.png`, `tokenizer_fertility.csv/png`, `output_mix.csv`, `error_types.csv/png` |
| Efficiency | `figures/time_vs_factuality.png`, `efficiency.tex` |
| Human evaluation | `07_human/human_eval_report.md` and its CSVs (or `07_llm_judge/llm_judge_report.md` for LLM-judge labels) |
| Qualitative examples | `06_evaluate/examples.json` (articles where the mono summary is clean but the code-mixed one is not, with the Track B/BV outputs) |
| One-paragraph summary of all tests | `06_evaluate/summary.md` |

### Metrics and the three objectives

| Objective | Metrics |
|---|---|
| Content preservation | ROUGE-1/2/L (P/R/F1), METEOR, entity and numeral recall/F1 against the reference |
| Meaning preservation | BERTScore (P/R/F1, mBERT layer 9), LaBSE similarity to the reference |
| Factual consistency | NLI support and hallucination rate, QA consistency F1, entity/numeral/date precision against the source, "any hallucination" rate |
| Efficiency and structure | average summary length, compression ratio (input words ÷ summary words), inference time per summary, empty outputs |

Rates and 0–1 scores are reported as percentages. All comparisons use paired tests on the same articles: Wilcoxon signed-rank for scores and exact McNemar for the binary hallucination flags. Each comes with a 95% bootstrap confidence interval, and p-values are Holm-corrected.

**Notes on reading the numbers**

* Our ROUGE has no stemming, so it is lower than published XL-Sum numbers. Notebook 06 also tries to install the official XL-Sum scorer; if that works, the `xlsum_rouge*` columns are comparable with the XL-Sum paper.
* METEOR uses its exact-match module only, because its synonym and stem modules are English-only.
* Inference time for the LLM is per GPU under batched decoding: batch time divided by batch size. Track BV's time includes the check and the revision.

## 7. The research paper (IEEE, Word + PDF)

`factindic/paper.py` writes the IEEE two-column paper as a Word file. Every section is written: abstract,
introduction, related work, problem, data, systems, verification, setup, results, discussion, limitations,
conclusion, and 38 references.

* **Before the run**, `python -m factindic.paper --out FactIndic_paper_draft.docx` gives a template. All numbers,
  tables and figures are yellow placeholders.
* **After the run**, notebook 06 writes `06_evaluate/FactIndic_paper_draft.docx` and notebook 07 writes
  `FactIndic_paper_draft.docx` in the output folder. In these drafts the numbers in the text, all nine tables and Figs. 2–5
  are filled in from your results. Outside Kaggle, run `python -m factindic.paper --results <factindic_out> --out paper.docx`.
* The sentences that **interpret** the results are yellow `[Write: …]` notes, and the team writes them.
  Also fill the GPU-hours placeholder in Section VII-D.
* For the PDF, open the file in Word and use File → Save As → PDF, or run `soffice --headless --convert-to pdf paper.docx`.
  Hindi and Bengali text needs a font that covers those scripts. On Windows, Nirmala UI is set by default.

## 8. Suggested team split

| Member | Owns | Paper sections |
|---|---|---|
| 1 | Notebooks 01–02, the code-mix quality check | Dataset, code-mixed variant construction |
| 2 | Notebook 03 (mT5, IndicBART, LoRA) | Baselines |
| 3 | Notebook 04, prompt design (Tracks A/B/BV) | Method: prompting strategies |
| 4 | Notebooks 05–06, tables and figures | Verification layer, results, code-mixing analysis |
| 5 | Human evaluation (recruit annotators, distribute sheets, notebook 07), integration | Human evaluation, limitations, abstract/conclusion |

Annotators need to read Hindi or Bengali fluently. Recruit Bengali readers early if the team has none.

## 9. Running outside Kaggle

```bash
pip install -r requirements.txt            # plus torch for your machine
python -m factindic all --profile mock     # CPU dry run of stages 01–06 into ./factindic_out
python -m factindic 04 --profile full --set n_test=50 summarizer_model=Qwen/Qwen3-4B-Instruct-2507
pytest -q tests/                           # unit tests + full chain with stand-in and tiny random models
```

`tests/run_notebooks_kaggle_sim.py` executes all seven notebooks in a simulated Kaggle folder layout.

## 10. Limitations to state in the report

* The code-mixed test inputs are LLM-generated, not naturally occurring. The validation checks and the human check limit this but do not remove it. The natural set, which is small, is the ecological check.
* The checkers are automatic. NLI and QA models make errors, especially on Bengali, which has no XNLI test set. This is why the human evaluation reports each checker's precision and recall.
* Code-mixing written as English words in Devanagari or Bengali script (पुलिस, কোর্ট) is not detected by the script-based CMI.
* Numbers written as words (दो, তিন) are not checked; only digits are.
* XL-Sum reference summaries are BBC intro sentences. They sometimes contain facts that are not in the article body, which lowers the reference-based scores equally across conditions.

## 11. Main references

Hasan et al. 2021 (XL-Sum) · Dabre et al. 2022 (IndicBART) · Xue et al. 2021 (mT5) · Hu et al. 2022 (LoRA) · Laban et al. 2022 (SummaC) · Fabbri et al. 2022 (QAFactEval) · Wang et al. 2020 (QAGS) · Nan et al. 2021 (entity-level factual consistency) · Laurer et al. 2022 (multilingual NLI) · Mhaske et al. 2023 (IndicNER/Naamapadam) · Das & Gambäck 2014 (Code-Mixing Index) · Zhang et al. 2020 (BERTScore) · Feng et al. 2022 (LaBSE) · Banerjee & Lavie 2005 (METEOR) · Lin 2004 (ROUGE) · Qwen Team 2025 (Qwen3) · Srivastava & Singh 2021 (HinGE, Hinglish generation and evaluation) · Mehnaz et al. 2021 (GupShup).
