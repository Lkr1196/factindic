"""Unit tests for the text processing, factuality checks, metrics and statistics.

    pytest -q tests/test_core.py
"""
import math
from pathlib import Path

import numpy as np
import pytest

from factindic import prompts
from factindic.codemix import cmi_stats, parse_numbered, validate_rewrite
from factindic.entities import (KeyIndex, dates_supported, entity_in, extract_dates, extract_numbers,
                                numbers_supported, same_entity)
from factindic.metrics import meteor, rouge
from factindic.stats import cohen_kappa, diff_in_diff, holm, paired_test, prf_binary
from factindic.textnorm import beng_to_deva, deva_to_beng, normalize_digits, split_sentences, tokenize
from factindic.verify import alignment, answer_f1, premises_for


def test_tokenize_keeps_matras_and_numbers():
    toks = tokenize("भारत में प्रदूषण का स्तर 6.2 प्रतिशत बढ़ा, 1,50,000 लोग।")
    assert "प्रदूषण" in toks and "बढ़ा" in toks
    assert "6.2" in toks and "1,50,000" in toks


def test_digits():
    assert normalize_digits("२०२३ ও ২০২৪") == "2023 ও 2024"


def test_sentence_split():
    s = split_sentences("पहला वाक्य है। दूसरा वाक्य 6.2 प्रतिशत है. Dr. A. P. J. Kalam ने कहा। ঠিক আছে?")
    assert len(s) == 4
    assert s[1].startswith("दूसरा") and "6.2" in s[1]


def test_numbers_with_indian_scales():
    vals = {round(n.value) for n in extract_numbers("5 लाख लोग, 2.5 करोड़ रुपये, 1,50,000 টাকা, ২ কোটি, 3 crore")}
    assert {500000, 25000000, 150000, 20000000, 30000000} <= vals
    kinds = {n.raw: n.kind for n in extract_numbers("2023 में 6.2% और 45 प्रतिशत")}
    assert kinds == {"2023": "year", "6.2": "percent", "45": "percent"}


def test_numbers_supported_and_rounding():
    sup, uns = numbers_supported("महंगाई 6.2% रही और 2024 में", "महंगाई 6.23 प्रतिशत रही, साल 2023")
    assert [n.raw for n in sup] == ["6.2"] and [n.raw for n in uns] == ["2024"]
    # same value written with a scale word
    sup, uns = numbers_supported("50 हज़ार लोग", "50,000 लोग")
    assert len(sup) == 1 and not uns


def test_dates():
    assert extract_dates("15 मार्च को और 3 জুনে") == [(15, 3), (3, 6)]
    sup, uns = dates_supported("16 मार्च को", "15 मार्च को")
    assert uns == [(16, 3)]


@pytest.mark.parametrize("a,b", [("दिल्ली", "Delhi"), ("मुंबई", "Mumbai"), ("कांग्रेस", "Congress"), ("भाजपा", "BJP"),
                                 ("কলকাতায়", "Kolkata"), ("राहुल गांधी", "Rahul Gandhi"), ("नरेंद्र मोदी", "मोदी"),
                                 ("বাংলাদেশ", "Bangladesh"), ("ভারত", "ভারতের")])
def test_cross_script_entities_match(a, b):
    assert same_entity(a, b)


@pytest.mark.parametrize("a,b", [("दिल्ली", "मुंबई"), ("Modi", "Mumbai"), ("पटना", "लखनऊ")])
def test_cross_script_entities_differ(a, b):
    assert not same_entity(a, b)


def test_entity_in_source():
    idx = KeyIndex("Narendra Modi ने दिल्ली में कहा")
    assert entity_in("नरेंद्र मोदी", idx)
    assert not entity_in("राहुल गांधी", idx)


def test_cmi():
    mono = cmi_stats("भारत में प्रदूषण का स्तर लगातार बढ़ रहा है।", "hi")
    cm = cmi_stats("भारत में pollution का level लगातार increase हो रहा है।", "hi")
    roman = cmi_stats("Bharat mein pollution ka level lagatar increase ho raha hai.", "hi", romanized=True)
    assert mono["cmi"] == 0
    assert 20 < cm["cmi"] < 50
    assert abs(roman["cmi"] - cm["cmi"]) < 15


def test_validate_rewrite():
    src = "भारत में 2023 में प्रदूषण का स्तर 6.2 प्रतिशत बढ़ा।"
    assert validate_rewrite(src, "भारत में 2023 में pollution का level 6.2 percent बढ़ा।", "cm", "hi")["ok"]
    assert "numbers changed" in validate_rewrite(src, "भारत में 2024 में pollution का level 6.2 percent बढ़ा।", "cm", "hi")["hard"]
    assert validate_rewrite(src, "Bharat mein 2023 mein pollution ka level 6.2 percent badha.", "roman", "hi")["ok"]
    assert not validate_rewrite(src, "Bharat mein 2023 में pollution ka level 6.2 percent badha.", "roman", "hi")["ok"]


def test_parsers():
    assert parse_numbered("1. पहला\n2) दूसरा\nnoise\n3: तीसरा", 3) == {1: "पहला", 2: "दूसरा", 3: "तीसरा"}
    assert prompts.parse_summary("FACTS:\n- a\nENTITIES: x\nSUMMARY: यह सारांश है।") == "यह सारांश है।"
    assert prompts.parse_summary("<think>hmm</think>\nयह सारांश है।") == "यह सारांश है।"
    assert prompts.parse_selfcheck("1. SUPPORTED\n2. UNSUPPORTED: wrong year", 2) == {1: None, 2: "wrong year"}
    assert prompts.parse_qg("Q: कौन? || A: मोदी\nQ: कब? || A: 2023") == [("कौन?", "मोदी"), ("कब?", "2023")]


def test_prompts_contain_article_and_number_lock():
    msgs = prompts.summarize_B("दिल्ली में 5 लाख लोग 15 मार्च को।", "hi", "cm", 40)
    user = msgs[-1]["content"]
    assert "5, 15" in user or ("5" in user and "15" in user)
    assert "SUMMARY:" in user and prompts.ART_OPEN in user


def test_rouge_matches_rouge_score_with_same_tokenizer():
    rouge_scorer = pytest.importorskip("rouge_score.rouge_scorer")

    class Tok:
        def tokenize(self, t):
            return tokenize(t)

    try:
        sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=Tok())
    except TypeError:
        pytest.skip("rouge_score variant without tokenizer argument")
    p, r = "भारत में प्रदूषण का स्तर बढ़ा", "भारत में प्रदूषण बहुत तेज़ी से बढ़ा"
    ref = sc.score(r, p)
    ours = rouge(p, r)
    assert math.isclose(ours["rouge1_f"], 100 * ref["rouge1"].fmeasure, rel_tol=1e-6)
    assert math.isclose(ours["rouge2_f"], 100 * ref["rouge2"].fmeasure, rel_tol=1e-6)
    assert math.isclose(ours["rougeL_f"], 100 * ref["rougeL"].fmeasure, rel_tol=1e-6)


def test_meteor_bounds():
    assert meteor("भारत में प्रदूषण", "भारत में प्रदूषण") > 90
    assert meteor("कुछ नहीं", "भारत में प्रदूषण") == 0


def test_alignment_and_answer_f1():
    al = alignment("दिल्ली में 6 लाख लोग प्रभावित हुए, राहुल ने कहा", "दिल्ली में 5 लाख लोग प्रभावित हुए।",
                   "दिल्ली में 5 लाख लोग प्रभावित", ["दिल्ली", "राहुल"], ["दिल्ली"])
    assert al["unsupported_nums"] == ["6"] and al["unsupported_ents"] == ["राहुल"]
    assert al["ent_prec_src"] == 0.5 and al["ent_rec_ref"] == 1.0
    assert al["fact_halluc"] == 1.0
    assert answer_f1("Delhi", "दिल्ली") == 1.0
    assert answer_f1("5 लाख", "500000") == 1.0
    assert answer_f1("2023", "2024") == 0.0
    assert len(premises_for(["a", "b", "c", "d"])) == 4 + 2


def test_script_conversion_roundtrip():
    bn = "কলকাতায় বিকাশ"
    assert deva_to_beng(beng_to_deva(bn)) == bn


def test_stats():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.1, 80)
    b = a - 0.1 + rng.normal(0, 0.02, 80)
    t = paired_test(a, b, 500)
    assert t["diff"] < 0 and t["ci_hi"] < 0 and t["p"] < 1e-6
    tb = paired_test(np.zeros(30), np.r_[np.ones(10), np.zeros(20)], 200, binary=True)
    assert tb["test"] == "McNemar" and tb["n01"] == 10 and tb["p"] < 0.01
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    d = diff_in_diff(a, a - 0.2, a, a - 0.1, 300)
    assert d["did"] == pytest.approx(0.1, abs=1e-9)
    assert cohen_kappa(["S", "H", "S", "P"], ["S", "H", "S", "P"]) == 1.0
    assert prf_binary([1, 0, 1, 1], [1, 1, 0, 1])["f1"] == pytest.approx(2 / 3)


def test_mock_pipeline(tmp_path, monkeypatch):
    """The whole chain on CPU with stand-in models."""
    monkeypatch.setenv("FACTINDIC_OUT", str(tmp_path / "out"))
    from factindic import Config
    from factindic import pipeline as P

    cfg = Config.make("mock")
    P.run_stage01(cfg)
    var = P.run_stage02(cfg)
    assert set(var["variant"]) >= {"mono", "cm", "roman"}
    P.run_stage03(cfg)
    llm = P.run_stage04(cfg)
    assert set(llm["track"]) == {"A", "B", "BV"}
    ver = P.run_stage05(cfg)
    assert ver["nli_score"].notna().mean() > 0.9
    res = P.run_stage06(cfg)
    assert (tmp_path / "out" / "06_evaluate" / "tables" / "h1_codemix_gap.csv").exists()
    assert (tmp_path / "out" / "06_evaluate" / "figures" / "nli_support_by_input.png").exists()
    assert len(res) == len(ver)
    from factindic.paper import build_paper

    filled = build_paper(str(tmp_path / "out"), str(tmp_path / "paper.docx"))
    template = build_paper(None, str(tmp_path / "template.docx"))
    assert Path(filled).stat().st_size > 50_000 and Path(template).exists()


def test_choose_quantization():
    from factindic.llm import choose_quantization

    assert choose_quantization("Qwen/Qwen3-8B", "auto", total_gpu_gb=15.8) == "4bit"      # Colab, one T4
    assert choose_quantization("Qwen/Qwen3-8B", "auto", total_gpu_gb=31.6) is None        # Kaggle, two T4s
    assert choose_quantization("Qwen/Qwen3-4B-Instruct-2507", "auto", total_gpu_gb=15.8) is None
    assert choose_quantization("Qwen/Qwen3-8B", "none", total_gpu_gb=15.8) is None
    assert choose_quantization("x/y-1B", "8bit", total_gpu_gb=80) == "8bit"
    assert choose_quantization("Qwen/Qwen3-8B", "auto", total_gpu_gb=0) is None            # CPU


def test_generation_resumes_after_interruption(tmp_path, monkeypatch):
    """A stage interrupted half-way (e.g. a Colab disconnect) continues from the cached results."""
    from factindic import Config, llm, prompts
    from factindic.mock_models import MockLLM

    calls = []

    class Counting(MockLLM):
        def generate(self, items, log=print):
            calls.append(len(items))
            return super().generate(items, log)

    monkeypatch.setattr(llm, "load_backend", lambda kind, model_id, cfg: Counting())
    cfg = Config.make("mock", out_root=str(tmp_path))
    items = [{"id": f"x{i}", "messages": prompts.summarize_A(f"दिल्ली में {i} लोग।", "hi", "mono", 40),
              "max_new_tokens": 20, "task": "sumA"} for i in range(10)]
    first = llm.run_generation(items[:6], "m", cfg, tag="t")      # "interrupted" after 6 items
    full = llm.run_generation(items, "m", cfg, tag="t")           # re-run of the whole stage
    assert calls == [6, 4]
    assert [r["id"] for r in full] == [it["id"] for it in items]
    assert [r["text"] for r in full[:6]] == [r["text"] for r in first]
    from factindic.handoff import cleanup_jobs

    cleanup_jobs(cfg)
    assert not (tmp_path / "_cache").exists()


def _fake_xlsum_rows(n, split):
    return [{"id": f"{split}-{i}", "url": "", "title": "t", "summary": "सारांश " * 10,
             "text": "यह एक वाक्य है। " * 60} for i in range(n)]


def test_load_xlsum_from_parquet_branch(tmp_path, monkeypatch):
    import huggingface_hub
    import pandas as pd

    from factindic import data

    for split, n in (("test", 3), ("train", 5)):
        d = tmp_path / "hindi" / split
        d.mkdir(parents=True)
        pd.DataFrame(_fake_xlsum_rows(n, split)).to_parquet(d / "0000.parquet")
    seen = {}

    def lrf(repo, repo_type=None, revision=None):
        seen["rev"] = revision
        assert revision == "refs/convert/parquet"
        return ["README.md", "hindi/test/0000.parquet", "hindi/train/0000.parquet"]

    monkeypatch.setattr(huggingface_hub, "list_repo_files", lrf)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, f, repo_type=None, revision=None: str(tmp_path / f))
    df = data.load_xlsum("hi", "test")
    assert len(df) == 3 and seen["rev"] == "refs/convert/parquet"


def test_load_xlsum_from_archive(tmp_path, monkeypatch):
    import io
    import json as _json
    import tarfile

    import huggingface_hub

    from factindic import data

    arch = tmp_path / "bengali_XLSum_v2.0.tar.bz2"
    with tarfile.open(arch, "w:bz2") as tf:
        for split, suffix in (("train", "_train"), ("validation", "_val"), ("test", "_test")):
            blob = "\n".join(_json.dumps(r, ensure_ascii=False) for r in _fake_xlsum_rows(4, split)).encode()
            info = tarfile.TarInfo(f"bengali/bengali{suffix}.jsonl")
            info.size = len(blob)
            tf.addfile(info, io.BytesIO(blob))

    def lrf(repo, repo_type=None, revision=None):
        if revision:
            raise RuntimeError("revision not found")
        return ["xlsum.py", "data/bengali_XLSum_v2.0.tar.bz2", "data/hindi_XLSum_v2.0.tar.bz2"]

    monkeypatch.setattr(huggingface_hub, "list_repo_files", lrf)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, f, repo_type=None, revision=None: str(arch))
    monkeypatch.setenv("FACTINDIC_CACHE", str(tmp_path / "cache"))
    val = data.load_xlsum("bn", "validation")
    assert len(val) == 4 and val["id"].iloc[0].startswith("validation")
    assert len(data.load_xlsum("bn", "test")) == 4    # second split comes from the extracted cache


def test_finalize_drops_half_sentence_at_token_limit():
    from factindic.summarize import _finalize

    assert _finalize("राम घर गया। वह खुश था और उस", 40, True) == ("राम घर गया।", True)
    assert _finalize("राम घर गया। वह खुश था।", 40, True) == ("राम घर गया। वह खुश था।", False)
    assert _finalize("राम घर गया और उस", 40, True) == ("राम घर गया और उस", True)  # nothing complete to keep
    assert _finalize("राम घर गया और उस", 40, False) == ("राम घर गया और उस", False)
    long = " ".join(["शब्द"] * 100) + "।"
    out, cut = _finalize(long, 40, False)
    assert cut and len(out.split()) <= 80


def test_token_budgets_fit_indic_summaries():
    from factindic.config import Config

    cfg = Config.make("full")
    # about 7 Qwen tokens per Bengali word: a 40-word summary needs 280 tokens
    assert cfg.max_new_tokens_A >= 7 * cfg.summary_max_words * 1.4
    assert cfg.max_new_tokens_B >= 3 * cfg.max_new_tokens_A


def test_llm_judge_labels_are_never_reported_as_human(tmp_path, monkeypatch):
    import glob
    import json

    import pandas as pd

    from factindic import pipeline as P
    from factindic.annotate import apply_judge_labels
    from factindic.config import Config
    from factindic.paper import Results

    monkeypatch.setenv("FACTINDIC_IN", "")
    cfg = Config.make("mock", out_root=str(tmp_path / "out"))
    for fn in (P.run_stage01, P.run_stage02, P.run_stage03, P.run_stage04, P.run_stage05, P.run_stage06):
        fn(cfg, log=lambda *a: None)
    sheets = tmp_path / "out" / "06_evaluate" / "human_eval"
    key = pd.read_csv(sheets / "human_eval_KEY_do_not_share.csv")
    units = key.drop_duplicates(["summary_id", "sent_idx"])
    labels = pd.DataFrame({"summary_id": units["summary_id"], "sent_idx": units["sent_idx"],
                           "label": ["Hallucinated" if i % 4 == 0 else "Supported" for i in range(len(units))],
                           "error_type": ["Entity" if i % 4 == 0 else None for i in range(len(units))],
                           "fluency": 4, "coverage": 3, "comment": ""})
    files = apply_judge_labels(str(sheets), labels, str(tmp_path / "judge"), "Test judge")
    assert files and json.loads((tmp_path / "judge" / "judge.json").read_text())["judge"] == "llm"
    out = P.run_stage07(cfg, sorted(glob.glob(str(tmp_path / "judge" / "human_eval_*_annotator*.xlsx"))), log=lambda *a: None)
    assert out["judge"] == "llm" and len(out["agreement"]) == 0 and out["n_labels"] == len(units)
    st = tmp_path / "out" / "07_llm_judge"
    assert (st / "judge_rates.csv").exists() and (st / "llm_judge_report.md").exists()
    assert not (tmp_path / "out" / "07_human").exists() and not (st / "agreement.csv").exists()
    assert "human" not in (st / "llm_judge_report.md").read_text(encoding="utf-8").replace("not by human annotators", "").lower()
    R = Results(str(tmp_path / "out"))
    assert R.llm_judge and R.judge_name == "Test judge" and R.agree is None and "human_halluc_rate" in R.hrates
