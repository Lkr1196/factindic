# Executed notebooks

These are the seven FactIndic notebooks as they ran on Kaggle, downloaded after each run with their full output. They are not edited. The clean versions to run yourself are one folder up, in `notebooks/`.

| Notebook | Run on | Hardware | Run time | What the run produced |
|---|---|---|---|---|
| [01 Data preparation](01_data_preparation_run.ipynb) | 12 Nov 2026 | CPU | 31 min | XL-Sum Hindi and Bengali articles cleaned, filtered and sampled (200 test, 30 dev, 150 training per language), plus 133 recent news articles collected from RSS feeds |
| [02 Code-mixed variants](02_codemix_generation_run.ipynb) | 13 Nov 2026 | T4 GPU | 3 h 36 min | Code-mixed and romanised versions of every article, checked sentence by sentence |
| [03 Fine-tuned summarizers](03_seq2seq_baselines_run.ipynb) | 15 Nov 2026 | T4 GPU | 32 min | mT5 and IndicBART summaries of every input, and the LoRA-adapted mT5 (dev loss 2.67 → 2.33) |
| [04 LLM summarization](04_llm_summarization_run.ipynb) | 15 Nov 2026 | T4 GPU | 8 h 1 min | Qwen3-4B summaries with Tracks A, B and BV, and the questions for the QA check |
| [05 Factuality verification](05_factuality_verification_run.ipynb) | 16 Nov 2026 | T4 GPU | 35 min | NLI, QA and entity, number and date checks of 7,998 summaries |
| [06 Evaluation and analysis](06_evaluation_analysis_run.ipynb) | 16 Nov 2026 | T4 GPU | 6 min | All metrics, significance tests, tables and figures, and the annotation sheets |
| [07 LLM-judge analysis](07_human_evaluation_run.ipynb) | 21 Nov 2026 | CPU | 1 min | Analysis of the LLM-judge labels and the paper draft |

Total run time: about 13 h 20 min.

Notebook 07 is written for human annotation sheets. In this run it read LLM-judge labels, as its output states, and all its results are reported as LLM-judge evaluation.
