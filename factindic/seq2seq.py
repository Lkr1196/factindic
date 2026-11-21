"""Fine-tuned seq2seq summarisers (notebook 03): mT5 (XL-Sum), IndicBART (XL-Sum), and a LoRA
adapter that adapts mT5 to code-mixed input.

* ``csebuetnlp/mT5_multilingual_XLSum``  – mT5-base fine-tuned on all 45 XL-Sum languages.
* ``ai4bharat/IndicBART-XLSum``           – IndicBART fine-tuned on the Indic part of XL-Sum. It reads
  every language in Devanagari, so Bengali input is converted to Devanagari and the output back.
* LoRA (``mt5-lora-cm``)                  – a low-rank adapter on the mT5 checkpoint trained on the
  training articles in all three variants (mono + cm + roman) with the XL-Sum reference summaries.
  It tests whether a little exposure to code-mixed input closes the gap.

mT5 is run in fp32: T5-family models overflow in fp16 (T4 GPUs have no bf16).
"""
from __future__ import annotations

import math
import random
import re
import time

from .textnorm import beng_to_deva, deva_to_beng, normalize_text


def _ws(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\n+", " ", (text or "").strip()))



def _peft_ignore_torchao():
    """Kaggle ships an old torchao; peft then raises ImportError as soon as it adds a LoRA layer.

    torchao is only needed for torchao-quantised models, which we do not use, so tell peft it is absent.
    """
    try:
        import peft.import_utils as iu
        import peft.tuners.lora.torchao as lt

        iu.is_torchao_available = lambda: False
        lt.is_torchao_available = lambda: False
    except Exception:
        pass


class Seq2SeqSummarizer:
    def __init__(self, kind: str, model_id: str, device: str | None = None, batch_size: int = 8,
                 adapter_dir: str | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.kind = kind
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if kind == "indicbart":
            try:
                self.tok = AutoTokenizer.from_pretrained(model_id, do_lower_case=False, use_fast=False, keep_accents=True)
            except Exception:
                self.tok = AutoTokenizer.from_pretrained(model_id, do_lower_case=False, keep_accents=True)
        else:
            self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        if adapter_dir:
            from peft import PeftModel

            _peft_ignore_torchao()

            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.to(self.device).eval()
        self.batch_size = batch_size

    def _ids(self, tok_str: str) -> int:
        tid = self.tok.convert_tokens_to_ids(tok_str)
        if tid is None or tid == self.tok.unk_token_id:
            raise ValueError(f"token {tok_str} not in vocabulary")
        return tid

    def summarize(self, texts: list[str], lang: str) -> list[dict]:
        torch = self.torch
        out = [None] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            batch = [_ws(texts[i]) for i in idx]
            t0 = time.time()
            with torch.inference_mode():
                if self.kind == "indicbart":
                    tag = "<2hi>" if lang == "hi" else "<2bn>"
                    if lang == "bn":
                        batch = [beng_to_deva(b) for b in batch]
                    enc = self.tok([f"{b} </s> {tag}" for b in batch], add_special_tokens=False, return_tensors="pt",
                                   padding=True, truncation=True, max_length=1024).to(self.device)
                    gen = self.model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                              num_beams=4, max_length=84, min_length=1, no_repeat_ngram_size=3,
                                              early_stopping=True, pad_token_id=self._ids("<pad>"),
                                              bos_token_id=self._ids("<s>"), eos_token_id=self._ids("</s>"),
                                              decoder_start_token_id=self._ids(tag))
                else:
                    enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                    gen = self.model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                              max_length=84, no_repeat_ngram_size=2, num_beams=4)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.time() - t0
            for i, g in zip(idx, gen):
                text = self.tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                if self.kind == "indicbart":
                    text = re.sub(r"<2\w\w>", "", text).strip()
                    if lang == "bn":
                        text = deva_to_beng(text)
                out[i] = {"summary": normalize_text(text), "time_s": dt / len(idx), "batch_size": len(idx),
                          "n_out": int((g != self.tok.pad_token_id).sum())}
        return out


def train_lora(pairs: list[tuple[str, str]], dev_pairs: list[tuple[str, str]], cfg, out_dir: str, log=print) -> dict:
    """LoRA fine-tuning of the mT5 XL-Sum checkpoint on (article, reference summary) pairs."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    _peft_ignore_torchao()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.mt5_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.mt5_model)
    lcfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=cfg.lora_r, lora_alpha=2 * cfg.lora_r, lora_dropout=0.05,
                      target_modules=["q", "k", "v", "o"])
    model = get_peft_model(model, lcfg).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"LoRA: {n_train / 1e6:.2f}M trainable parameters, {len(pairs)} training pairs")

    def batches(data, bs, shuffle):
        data = list(data)
        if shuffle:
            random.Random(cfg.seed).shuffle(data)
        for i in range(0, len(data), bs):
            chunk = data[i:i + bs]
            enc = tok([_ws(a) for a, _ in chunk], return_tensors="pt", padding=True, truncation=True, max_length=512)
            lab = tok(text_target=[_ws(b) for _, b in chunk], return_tensors="pt", padding=True, truncation=True,
                      max_length=84)["input_ids"]
            lab[lab == tok.pad_token_id] = -100
            yield {k: v.to(device) for k, v in enc.items()}, lab.to(device)

    bs, accum = 4, 4
    steps = math.ceil(len(pairs) / bs) * cfg.lora_epochs // accum
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lora_lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, max(1, int(0.06 * steps)), max(1, steps))
    history = []
    t0 = time.time()
    for ep in range(cfg.lora_epochs):
        model.train()
        run, n = 0.0, 0
        for k, (enc, lab) in enumerate(batches(pairs, bs, True), 1):
            loss = model(**enc, labels=lab).loss / accum
            loss.backward()
            run += loss.item() * accum
            n += 1
            if k % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if k % 100 == 0:
                log(f"  epoch {ep + 1} step {k}: loss {run / n:.3f} ({(time.time() - t0) / 60:.1f} min)")
        model.eval()
        dl, dn = 0.0, 0
        with torch.inference_mode():
            for enc, lab in batches(dev_pairs, 8, False):
                dl += model(**enc, labels=lab).loss.item()
                dn += 1
        history.append({"epoch": ep + 1, "train_loss": run / max(1, n), "dev_loss": dl / max(1, dn)})
        log(f"  epoch {ep + 1}: train loss {run / max(1, n):.3f}, dev loss {dl / max(1, dn):.3f}")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return {"trainable_params": n_train, "history": history, "minutes": (time.time() - t0) / 60}
