"""Run every Hugging Face code path with tiny random models on CPU (no downloads).

    pytest -q tests/test_real_code_paths.py
"""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    spec = importlib.util.spec_from_file_location("mtm", Path(__file__).with_name("make_tiny_models.py"))
    mtm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mtm)
    mtm.main(str(root))
    return root


def test_full_chain_with_tiny_models(tiny, tmp_path, monkeypatch):
    monkeypatch.setenv("FACTINDIC_OUT", str(tmp_path / "out"))
    from factindic import Config
    from factindic import pipeline as P

    T = str(tiny)
    cfg = Config.make("quick", backend="hf", data_source="mock", summarizer_model=T + "/tiny-qwen",
                      codemix_model=T + "/tiny-qwen", mt5_model=T + "/tiny-mt5", indicbart_model=T + "/tiny-indicbart",
                      nli_model=T + "/tiny-nli", qa_model=T + "/tiny-qa", ner_model=T + "/tiny-ner",
                      bertscore_model=T + "/tiny-bert", bertscore_layers=2, labse_model=T + "/tiny-bert",
                      tokenizers_for_fertility=(T + "/tiny-qwen", T + "/tiny-mt5"), n_test=4, n_dev=2, n_train_cm=4,
                      qa_docs_per_lang=3, n_natural_per_lang=2, scrape=False, workers=0, n_bootstrap=50,
                      min_words=30, min_summary_words=5, human_docs_per_lang=3, max_new_tokens_A=16,
                      max_new_tokens_B=24)
    P.run_stage01(cfg)
    P.run_stage02(cfg)
    s2s = P.run_stage03(cfg)
    assert {"mt5", "indicbart", "mt5-lora-cm"} <= set(s2s["system"])
    llm = P.run_stage04(cfg)
    assert len(llm) > 0
    ver = P.run_stage05(cfg)
    assert len(ver) == len(s2s) + len(llm)
    res = P.run_stage06(cfg)
    assert "bertscore_f" in res.columns and "labse_ref" in res.columns


def test_worker_processes(tiny, tmp_path, monkeypatch):
    monkeypatch.setenv("FACTINDIC_OUT", str(tmp_path / "out"))
    from factindic import Config, prompts
    from factindic.llm import run_generation

    cfg = Config.make("quick", backend="hf", workers=2)
    items = [{"id": f"x{i}", "messages": prompts.summarize_A("भारत में " * (5 + i), "hi", "mono", 40),
              "max_new_tokens": 8, "task": "sumA"} for i in range(5)]
    out = run_generation(items, str(tiny / "tiny-qwen"), cfg, tag="t")
    assert [o["id"] for o in out] == [it["id"] for it in items]


def test_prompts_with_different_limits_share_a_batch(tiny):
    """Regression: grouping only identical max_new_tokens made every batch hold one prompt."""
    from factindic import prompts
    from factindic.llm import HFBackend

    b = HFBackend(str(tiny / "tiny-qwen"), max_batch_tokens=100_000, max_batch_size=8)
    items = [{"id": f"x{i}", "messages": prompts.codemix_rewrite(["भारत में प्रदूषण " * (2 + i)], "hi"),
              "max_new_tokens": 30 + 7 * i, "task": "cm"} for i in range(8)]
    out = b.generate(items)
    assert max(o["batch_size"] for o in out) == 8
    assert all(o["n_out"] <= it["max_new_tokens"] for o, it in zip(out, items))
    assert [o["id"] for o in out] == [it["id"] for it in items]
