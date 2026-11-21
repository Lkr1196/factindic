"""Build tiny, randomly initialised models with the same interfaces as the real ones.

Used by tests/test_real_code_paths.py to run every Hugging Face code path (generation with left
padding, NLI, extractive QA with overflow windows, token classification, mT5 + LoRA training,
IndicBART-style tagging, BERTScore, sentence-transformers) on CPU without downloads.
Outputs are random; only the plumbing is tested.

    python tests/make_tiny_models.py OUT_DIR
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CHAT_TEMPLATE = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
    "{% if enable_thinking is defined and enable_thinking is false %}{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}{% endif %}"
)


def corpus() -> list[str]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from factindic.mock_data import make_docs

    lines = []
    for lang in ("hi", "bn"):
        for d in make_docs(lang, 60, "corpus"):
            lines += [d["text"], d["summary"]]
    lines += ["The quick brown fox jumps over the lazy dog. Summary: pollution level increase government scheme.",
              "SUMMARY FACTS ENTITIES Q A SUPPORTED UNSUPPORTED number article sentence"] * 20
    return lines


def wordpiece_tokenizer(out: Path, texts: list[str]):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=1200, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tok.train_from_iterator(texts, trainer)
    cls, sep = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                                                       special_tokens=[("[CLS]", cls), ("[SEP]", sep)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]", model_max_length=512)
    fast.save_pretrained(out)
    return fast


def main(root: str) -> None:
    import torch
    from transformers import (BertConfig, BertForQuestionAnswering, BertForSequenceClassification,
                              BertForTokenClassification, BertModel)

    torch.manual_seed(0)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    texts = corpus()

    # ---- tiny BERT family (NLI / QA / NER / BERTScore / sentence embeddings)
    for name, cls, extra in (
        ("tiny-nli", BertForSequenceClassification, {"num_labels": 3, "id2label": {0: "entailment", 1: "neutral", 2: "contradiction"},
                                                     "label2id": {"entailment": 0, "neutral": 1, "contradiction": 2}}),
        ("tiny-qa", BertForQuestionAnswering, {}),
        ("tiny-ner", BertForTokenClassification, {"num_labels": 7, "id2label": {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC", 5: "B-ORG", 6: "I-ORG"},
                                                  "label2id": {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}}),
        ("tiny-bert", BertModel, {}),
    ):
        d = root / name
        tok = wordpiece_tokenizer(d, texts)
        cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=512, **extra)
        cls(cfg).save_pretrained(d)

    # ---- tiny causal LM with a Qwen-style chat template
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    bpe.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=1500, special_tokens=specials,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    ftok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                   additional_special_tokens=["<|im_start|>"])
    ftok.chat_template = CHAT_TEMPLATE
    d = root / "tiny-qwen"
    ftok.save_pretrained(d)
    qcfg = Qwen3Config(vocab_size=len(ftok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=8192,
                       eos_token_id=ftok.convert_tokens_to_ids("<|im_end|>"), pad_token_id=ftok.pad_token_id,
                       tie_word_embeddings=True)
    m = Qwen3ForCausalLM(qcfg)
    m.generation_config.eos_token_id = [ftok.convert_tokens_to_ids("<|im_end|>"), ftok.convert_tokens_to_ids("<|endoftext|>")]
    m.save_pretrained(d)

    # ---- sentencepiece model for mT5 and IndicBART
    import sentencepiece as spm

    sp_dir = root / "_spm"
    sp_dir.mkdir(exist_ok=True)
    (sp_dir / "corpus.txt").write_text("\n".join(texts), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(sp_dir / "corpus.txt"), model_prefix=str(sp_dir / "sp"), vocab_size=400,
                                   model_type="unigram", character_coverage=1.0, pad_id=0, eos_id=1, unk_id=2, bos_id=-1)
    from transformers import MT5Config, MT5ForConditionalGeneration, T5Tokenizer

    t5tok = T5Tokenizer(vocab_file=str(sp_dir / "sp.model"), extra_ids=0, legacy=True)
    d = root / "tiny-mt5"
    t5tok.save_pretrained(d)
    mcfg = MT5Config(vocab_size=t5tok.vocab_size, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_decoder_layers=2,
                     num_heads=4, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    MT5ForConditionalGeneration(mcfg).save_pretrained(d)

    from transformers import AlbertTokenizer, MBartConfig, MBartForConditionalGeneration

    atok = AlbertTokenizer(vocab_file=str(sp_dir / "sp.model"), do_lower_case=False, keep_accents=True,
                           bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
    atok.add_special_tokens({"additional_special_tokens": ["<2hi>", "<2bn>"]})
    for t in ("<s>", "</s>", "<pad>"):
        if atok.convert_tokens_to_ids(t) == atok.unk_token_id:
            atok.add_tokens([t], special_tokens=True)
    d = root / "tiny-indicbart"
    atok.save_pretrained(d)
    bcfg = MBartConfig(vocab_size=len(atok), d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                       decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, max_position_embeddings=1024,
                       pad_token_id=atok.convert_tokens_to_ids("<pad>"), bos_token_id=atok.convert_tokens_to_ids("<s>"),
                       eos_token_id=atok.convert_tokens_to_ids("</s>"), decoder_start_token_id=atok.convert_tokens_to_ids("<2hi>"))
    MBartForConditionalGeneration(bcfg).save_pretrained(d)
    print(json.dumps({p.name: sorted(x.name for x in p.iterdir())[:6] for p in root.iterdir()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny_models")
