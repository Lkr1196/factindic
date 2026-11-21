"""Project-wide configuration: languages, models, sample sizes and paths.

Edit values in the notebooks (``cfg = Config.make(profile=..., **overrides)``) rather than here.
Profiles:
  * ``full``  – the real experiment (default sizes below).
  * ``quick`` – a small run (about 25 documents per language) to check the whole chain on Kaggle.
  * ``mock``  – CPU-only dry run with stand-in models. No GPU or downloads. Use it to test the
                notebooks; never report its numbers.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

LANGS = {
    "hi": {
        "name": "Hindi", "native": "हिन्दी", "script": "Devanagari", "script_code": "deva",
        "xlsum": "hindi", "cm_name": "Hinglish", "indicbart_tag": "<2hi>",
        "cm_example_mono": "भारत में प्रदूषण का स्तर लगातार बढ़ रहा है।",
        "cm_example_cm": "भारत में pollution का level लगातार increase हो रहा है।",
        "roman_example": "Bharat mein pollution ka level lagatar increase ho raha hai.",
    },
    "bn": {
        "name": "Bengali", "native": "বাংলা", "script": "Bengali", "script_code": "beng",
        "xlsum": "bengali", "cm_name": "Banglish", "indicbart_tag": "<2bn>",
        "cm_example_mono": "শহরে বায়ু দূষণের মাত্রা দ্রুত বাড়ছে।",
        "cm_example_cm": "শহরে air pollution-এর level দ্রুত increase হচ্ছে।",
        "roman_example": "Shohore air pollution-er level druto increase hocche.",
    },
}

# Input conditions. "mono" is the original article; "cm" is the same article rewritten as
# code-mixed text in the native script + Latin English words; "roman" is the "cm" text fully
# romanised (WhatsApp-style Hinglish/Banglish). "natural" marks naturally code-mixed documents.
VARIANTS = ("mono", "cm", "roman")


def on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def default_out_root() -> str:
    env = os.environ.get("FACTINDIC_OUT")
    if env:
        return env
    return "/kaggle/working/factindic_out" if on_kaggle() else str(Path.cwd() / "factindic_out")


@dataclass
class Config:
    profile: str = "full"
    seed: int = 13
    langs: tuple = ("hi", "bn")
    variants: tuple = VARIANTS

    # ---------------- data ----------------
    data_source: str = "xlsum"   # xlsum | mock (template articles for dry runs)
    n_test: int = 200            # test articles per language (official XL-Sum test split)
    n_dev: int = 30              # dev articles per language (prompt sanity checks)
    n_train_cm: int = 60         # training articles per language turned code-mixed (for LoRA); 0 = skip
    min_words: int = 120         # article length filter (words, before truncation)
    max_words_raw: int = 1200
    lead_words: int = 300        # articles are cut to their first ~300 words at a sentence boundary
    max_sents: int = 14
    min_summary_words: int = 8
    max_summary_words: int = 60
    near_dup_threshold: float = 0.8
    natural_split: tuple = (0.7, 0.1, 0.2)  # article-level split for collected natural code-mixed docs
    scrape: bool = True          # try RSS scraping of Hindi/Bengali news (best effort)
    n_natural_per_lang: int = 100

    # ---------------- LLMs ----------------
    backend: str = "auto"        # auto | hf | vllm | openai | mock
    summarizer_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    codemix_model: str = "Qwen/Qwen3-4B-Instruct-2507"  # writes the code-mixed inputs; "Qwen/Qwen3-8B" is ~4x slower
    openai_base_url: str = ""    # e.g. https://api.groq.com/openai/v1 (only for backend=openai)
    openai_model: str = ""
    openai_concurrency: int = 4
    max_model_len: int = 6144
    max_batch_tokens: int = 24000  # HF backend: padded tokens per generation batch (lower if OOM)
    max_batch_size: int = 24
    repetition_penalty: float = 1.0  # >1 also penalises copying article tokens (names, numbers), so keep 1.0
    workers: int = -1            # -1 = one worker per GPU when the model fits on one GPU; 0 = in-process
    quantize: str = "auto"       # auto | none | 4bit | 8bit. auto = 4-bit only if the model does not fit in GPU memory

    # ---------------- summarisation ----------------
    summary_max_words: int = 40
    tracks: tuple = ("A", "B", "BV")   # A = standard, B = factuality-aware, BV = B + verify-and-revise
    # Token budgets. Qwen needs about 5 tokens per Hindi word and 7 per Bengali word, so a 40-word
    # summary can take 280 tokens; smaller budgets cut summaries off in the middle of a sentence.
    max_new_tokens_A: int = 448
    max_new_tokens_B: int = 1536
    seq2seq_models: tuple = ("mt5", "indicbart")
    mt5_model: str = "csebuetnlp/mT5_multilingual_XLSum"
    indicbart_model: str = "ai4bharat/IndicBART-XLSum"
    lora: bool = True
    lora_epochs: int = 2
    lora_lr: float = 3e-4
    lora_r: int = 16

    # ---------------- verification ----------------
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    nli_threshold: float = 0.5
    qa_model: str = "deepset/xlm-roberta-large-squad2"
    ner_model: str = "ai4bharat/IndicNER"
    qa_docs_per_lang: int = 100  # QA-based check runs on this many test articles per language
    qg_max_questions: int = 5

    # ---------------- evaluation ----------------
    bertscore_model: str = "bert-base-multilingual-cased"
    bertscore_layers: int = 9
    labse_model: str = "sentence-transformers/LaBSE"
    tokenizers_for_fertility: tuple = ("Qwen/Qwen3-4B-Instruct-2507", "csebuetnlp/mT5_multilingual_XLSum",
                                       "FacebookAI/xlm-roberta-base")
    n_bootstrap: int = 2000
    human_docs_per_lang: int = 30
    human_overlap: float = 0.25
    n_annotators: int = 3

    out_root: str = field(default_factory=default_out_root)

    # ------------------------------------------------------------------
    @classmethod
    def make(cls, profile: str = "full", **overrides) -> "Config":
        base = cls(profile=profile)
        if profile == "quick":
            base.n_test, base.n_dev, base.n_train_cm = 25, 5, 40
            base.qa_docs_per_lang, base.n_natural_per_lang = 15, 10
            base.human_docs_per_lang, base.lora_epochs = 6, 1
        elif profile == "mock":
            base.backend = "mock"
            base.data_source = "mock"
            base.n_test, base.n_dev, base.n_train_cm = 8, 2, 6
            base.qa_docs_per_lang, base.n_natural_per_lang = 6, 3
            base.human_docs_per_lang, base.lora_epochs = 4, 1
            base.n_bootstrap, base.scrape = 200, False
            base.min_words, base.min_summary_words = 30, 5
        elif profile != "full":
            raise ValueError(f"unknown profile {profile!r}")
        for k, v in overrides.items():
            if not hasattr(base, k):
                raise AttributeError(f"Config has no field {k!r}")
            setattr(base, k, v)
        return base

    @property
    def mock(self) -> bool:
        return self.backend == "mock"

    def path(self, *parts) -> Path:
        p = Path(self.out_root).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, default=list)

    def save(self, stage: str) -> None:
        self.path(stage, "config.json").write_text(self.to_json(), encoding="utf-8")


def lang_info(lang: str) -> dict:
    return LANGS[lang]
