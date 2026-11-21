"""LLM backends and a data-parallel runner.

Backends (all take the same "items" and return the same records):
  * ``hf``     – Hugging Face transformers, batched greedy decoding with left padding.
                 Works on Kaggle's preinstalled stack. Default.
  * ``vllm``   – vLLM, much faster, if you installed it (optional; see the README).
  * ``openai`` – any OpenAI-compatible HTTP API (Groq, Together, a local server, ...).
  * ``mock``   – deterministic stand-in for CPU dry runs.

An *item* is ``{"id": str, "messages": [...], "max_new_tokens": int, "task": str}``.
A *result* is ``{"id", "text", "n_in", "n_out", "time_s", "batch_size", "error"}``.

On a machine with two T4 GPUs and a model that fits on one GPU (e.g. a 4B model in fp16),
``run_generation`` starts one worker process per GPU and splits the items between them
(data parallelism, about 2x faster). Bigger models (8B) use one process spread over both GPUs.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

_BACKEND_CACHE: dict = {}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def gpu_mem_gb() -> float:
    """Memory of GPU 0 in GB, read with nvidia-smi so this process does not create a CUDA context."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return float(out[0]) / 1024
    except Exception:
        try:
            import torch

            return torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            return 0.0


def _dtype_kwarg() -> str:
    """transformers >= 4.56 renamed ``torch_dtype`` to ``dtype``."""
    try:
        import transformers
        from packaging.version import Version

        return "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    except Exception:
        return "torch_dtype"


def model_params_b(model_id: str) -> float:
    """Approximate parameter count in billions (from the Hub, else from the name)."""
    try:
        from huggingface_hub import model_info

        info = model_info(model_id)
        if getattr(info, "safetensors", None) and info.safetensors.total:
            return info.safetensors.total / 1e9
    except Exception:
        pass
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", model_id.split("/")[-1])
    return float(m.group(1)) if m else 8.0


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.replace("<think>", "").replace("</think>", "").strip()


def choose_quantization(model_id: str, quantize: str = "auto", total_gpu_gb: float | None = None) -> str | None:
    """None (fp16), '4bit' or '8bit'. 'auto' picks 4-bit only when fp16 weights would not fit on the
    visible GPUs, e.g. an 8B model on Colab's single T4 (on Kaggle's two T4s it fits in fp16)."""
    if quantize in ("4bit", "8bit"):
        return quantize
    if quantize == "none":
        return None
    if total_gpu_gb is None:
        try:
            import torch

            total_gpu_gb = sum(torch.cuda.get_device_properties(i).total_memory
                               for i in range(torch.cuda.device_count())) / 1e9
        except Exception:
            total_gpu_gb = 0.0
    if total_gpu_gb <= 0:
        return None
    need = model_params_b(model_id) * 2 * 1.15
    return "4bit" if need > 0.9 * total_gpu_gb else None


def chat_prompt(tokenizer, messages: list[dict]) -> str:
    """Apply the chat template with thinking disabled (Qwen3); fall back to plain text."""
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
    return "\n\n".join(parts) + "\n\nASSISTANT:"


# --------------------------------------------------------------------------------------
# Hugging Face backend
# --------------------------------------------------------------------------------------
class HFBackend:
    name = "hf"

    def __init__(self, model_id: str, max_batch_tokens: int = 24000, max_batch_size: int = 24,
                 repetition_penalty: float = 1.0, max_model_len: int = 6144, device_map=None, quantize: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        n_gpu = torch.cuda.device_count()
        kwargs = {_dtype_kwarg(): torch.float16 if n_gpu else torch.float32}
        if n_gpu:
            kwargs["device_map"] = device_map or "auto"
            q = choose_quantization(model_id, quantize)
            if q:
                from transformers import BitsAndBytesConfig

                print(f"  loading {model_id} in {q} (it does not fit in GPU memory in fp16)")
                kwargs["quantization_config"] = (
                    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                       bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
                    if q == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa", **kwargs)
        except (ValueError, TypeError):
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        if self.tok.eos_token_id is not None:
            self.eos_ids.add(self.tok.eos_token_id)
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
        self.repetition_penalty = repetition_penalty
        self.max_model_len = max_model_len
        self.last_batch_sizes: list[int] = []

    def _gen_config(self, max_new_tokens: int):
        from transformers import GenerationConfig

        return GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False,
                                repetition_penalty=self.repetition_penalty,
                                pad_token_id=self.tok.pad_token_id, eos_token_id=sorted(self.eos_ids))

    def _gen_batch(self, ids: list[list[int]], max_new_tokens: int):
        torch = self.torch
        pad = self.tok.pad_token_id
        L = max(len(x) for x in ids)
        input_ids = torch.tensor([[pad] * (L - len(x)) + x for x in ids], device=self.device)
        attn = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids], device=self.device)
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(input_ids=input_ids, attention_mask=attn,
                                      generation_config=self._gen_config(max_new_tokens))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        res = []
        for row in out[:, L:].tolist():
            toks = []
            for t in row:
                if t in self.eos_ids:
                    break
                toks.append(t)
            res.append((toks, strip_think(self.tok.decode(toks, skip_special_tokens=True))))
        return res, dt

    def generate(self, items: list[dict], log=print) -> list[dict]:
        """Batched greedy generation.

        Items with different ``max_new_tokens`` share a batch: the batch runs to the largest limit (each
        row still stops at its own end-of-text token) and every row is then cut to its own limit. Grouping
        only identical limits would make every batch hold a single prompt when the limits all differ.
        """
        torch = self.torch
        prompts = [chat_prompt(self.tok, it["messages"]) for it in items]
        enc = [self.tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        results: dict[int, dict] = {}
        for j, it in enumerate(items):
            budget = self.max_model_len - it["max_new_tokens"]
            if len(enc[j]) > budget:  # keep the end of the prompt (instructions come last)
                enc[j] = enc[j][:budget // 2] + enc[j][-(budget - budget // 2):]
        # similar output limits together (buckets of 128 tokens), longest prompts first (less padding)
        bucket = lambda j: -(-items[j]["max_new_tokens"] // 128)  # noqa: E731
        order = sorted(range(len(items)), key=lambda j: (bucket(j), -len(enc[j])))
        i = 0
        while i < len(order):
            first = order[i]
            window = [j for j in order[i:i + self.max_batch_size] if bucket(j) == bucket(first)] or [first]
            mnt = max(items[j]["max_new_tokens"] for j in window)
            L = len(enc[first])
            bs = max(1, min(len(window), self.max_batch_tokens // (L + mnt)))
            batch = window[:bs]
            mnt = max(items[j]["max_new_tokens"] for j in batch)
            try:
                outs, dt = self._gen_batch([enc[j] for j in batch], mnt)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(batch) == 1:
                    results[first] = {"text": "", "n_out": 0, "time_s": 0.0, "batch_size": 1,
                                      "error": "CUDA out of memory"}
                    log(f"  OOM even for a single prompt ({L} prompt tokens); skipped it")
                    i += 1
                    continue
                self.max_batch_tokens = max(2000, int(self.max_batch_tokens * 0.7))
                log(f"  OOM with {len(batch)} prompts: lowering max_batch_tokens to {self.max_batch_tokens}")
                continue
            for j, (toks, text) in zip(batch, outs):
                cap = items[j]["max_new_tokens"]
                if len(toks) > cap:  # this row ran past its own limit while the batch was finishing
                    toks = toks[:cap]
                    text = strip_think(self.tok.decode(toks, skip_special_tokens=True))
                results[j] = {"text": text, "n_out": len(toks), "time_s": dt / len(batch),
                              "batch_size": len(batch), "error": None}
            self.last_batch_sizes.append(len(batch))
            i += len(batch)
        return [{"id": items[j]["id"], "n_in": len(enc[j]), **results[j]} for j in range(len(items))]


# --------------------------------------------------------------------------------------
# vLLM backend (optional)
# --------------------------------------------------------------------------------------
class VLLMBackend:
    name = "vllm"

    def __init__(self, model_id: str, max_model_len: int = 6144, repetition_penalty: float = 1.0,
                 tensor_parallel_size: int | None = None, gpu_memory_utilization: float = 0.88):
        from vllm import LLM

        tp = tensor_parallel_size or max(1, gpu_count())
        self.llm = LLM(model=model_id, dtype="half", tensor_parallel_size=tp, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, enforce_eager=True,
                       disable_custom_all_reduce=True, seed=0)
        self.tok = self.llm.get_tokenizer()
        self.repetition_penalty = repetition_penalty

    def generate(self, items: list[dict], log=print) -> list[dict]:
        from vllm import SamplingParams

        prompts = [chat_prompt(self.tok, it["messages"]) for it in items]
        groups: dict[int, list[int]] = {}
        for j, it in enumerate(items):
            groups.setdefault(it["max_new_tokens"], []).append(j)
        res: dict[int, dict] = {}
        for mnt, idx in groups.items():
            sp = SamplingParams(temperature=0.0, max_tokens=mnt, repetition_penalty=self.repetition_penalty)
            t0 = time.time()
            outs = self.llm.generate([prompts[j] for j in idx], sp, use_tqdm=False)
            dt = time.time() - t0
            for j, o in zip(idx, outs):
                res[j] = {"text": strip_think(o.outputs[0].text), "n_in": len(o.prompt_token_ids),
                          "n_out": len(o.outputs[0].token_ids), "time_s": dt / len(idx),
                          "batch_size": len(idx), "error": None}
            log(f"  vLLM: {len(idx)} prompts in {dt:.0f}s")
        return [{"id": items[j]["id"], **res[j]} for j in range(len(items))]


# --------------------------------------------------------------------------------------
# OpenAI-compatible HTTP backend (optional)
# --------------------------------------------------------------------------------------
class OpenAIBackend:
    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str | None = None, concurrency: int = 4):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.concurrency = concurrency

    def _one(self, it: dict) -> dict:
        import requests

        body = {"model": self.model, "messages": it["messages"], "temperature": 0,
                "max_tokens": it["max_new_tokens"]}
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        delay = 2.0
        for attempt in range(8):
            t0 = time.time()
            try:
                r = requests.post(self.url, json=body, headers=headers, timeout=120)
                if r.status_code == 200:
                    data = r.json()
                    usage = data.get("usage") or {}
                    return {"id": it["id"], "text": strip_think(data["choices"][0]["message"]["content"] or ""),
                            "n_in": usage.get("prompt_tokens", 0), "n_out": usage.get("completion_tokens", 0),
                            "time_s": time.time() - t0, "batch_size": 1, "error": None}
                if r.status_code not in (408, 409, 429, 500, 502, 503, 504):
                    return {"id": it["id"], "text": "", "n_in": 0, "n_out": 0, "time_s": 0.0, "batch_size": 1,
                            "error": f"HTTP {r.status_code}: {r.text[:200]}"}
            except Exception as e:  # network error: retry
                err = str(e)
            time.sleep(delay)
            delay = min(60, delay * 2)
        return {"id": it["id"], "text": "", "n_in": 0, "n_out": 0, "time_s": 0.0, "batch_size": 1,
                "error": "gave up after retries"}

    def generate(self, items: list[dict], log=print) -> list[dict]:
        from concurrent.futures import ThreadPoolExecutor

        out = []
        with ThreadPoolExecutor(self.concurrency) as ex:
            for k, r in enumerate(ex.map(self._one, items), 1):
                out.append(r)
                if k % 50 == 0:
                    log(f"  {k}/{len(items)} done")
        return out


# --------------------------------------------------------------------------------------
# Backend factory and runner
# --------------------------------------------------------------------------------------
def load_backend(kind: str, model_id: str, cfg_dict: dict):
    if kind == "mock":
        from .mock_models import MockLLM

        return MockLLM()
    if kind == "openai":
        return OpenAIBackend(cfg_dict.get("openai_base_url", ""), cfg_dict.get("openai_model") or model_id,
                             os.environ.get("OPENAI_API_KEY"), cfg_dict.get("openai_concurrency", 4))
    if kind in ("auto", "vllm"):
        try:
            return VLLMBackend(model_id, max_model_len=cfg_dict.get("max_model_len", 6144),
                               repetition_penalty=cfg_dict.get("repetition_penalty", 1.0))
        except Exception as e:
            if kind == "vllm":
                raise
            if not isinstance(e, ImportError):
                print(f"  vLLM unavailable ({type(e).__name__}: {str(e)[:120]}); using transformers")
    return HFBackend(model_id, max_batch_tokens=cfg_dict.get("max_batch_tokens", 24000),
                     max_batch_size=cfg_dict.get("max_batch_size", 24),
                     repetition_penalty=cfg_dict.get("repetition_penalty", 1.0),
                     max_model_len=cfg_dict.get("max_model_len", 6144),
                     quantize=cfg_dict.get("quantize", "auto"))


def _plan_workers(model_id: str, cfg) -> int:
    n_gpu = gpu_count()
    if cfg.workers > 0:
        return cfg.workers if n_gpu == 0 else min(cfg.workers, n_gpu)
    if cfg.workers == 0 or n_gpu <= 1:
        return 0
    need_gb = model_params_b(model_id) * 2 * 1.25  # fp16 weights + working memory
    return n_gpu if need_gb < 0.8 * gpu_mem_gb() else 0


def progress_line(tag: str, done: int, total: int, elapsed: float, n_tokens: int, batch_sizes: list) -> str:
    """e.g. "rewrite_cm: 160/1450 generated (6.1 min, ~49 min left; avg batch 14.2, 512 tokens/s)"."""
    left = elapsed / max(1, done) * (total - done)
    avg_bs = sum(batch_sizes) / max(1, len(batch_sizes))
    return (f"  {tag}: {done}/{total} generated ({elapsed / 60:.1f} min, ~{left / 60:.0f} min left; "
            f"avg batch {avg_bs:.1f}, {n_tokens / max(elapsed, 1e-6):.0f} tokens/s)")


def _item_key(it: dict) -> str:
    import hashlib

    blob = json.dumps([it["messages"], it["max_new_tokens"]], ensure_ascii=False, sort_keys=True)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def _cache_file(cfg, model_id: str, tag: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{model_id}__{tag}")[-150:]
    p = Path(cfg.out_root) / "_cache" / f"{safe}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_cache(path: Path) -> dict:
    done = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["_key"]] = r
                except Exception:  # a half-written last line after a disconnect
                    pass
    return done


def run_generation(items: list[dict], model_id: str, cfg, tag: str = "gen") -> list[dict]:
    """Generate for all items. Returns results in the same order as ``items``.

    Results are appended to ``<out_root>/_cache`` as they finish, so if the session dies (e.g. a Colab
    disconnect) re-running the same stage resumes where it stopped. Each stage deletes the cache at the end.
    """
    if not items:
        return []
    from dataclasses import asdict

    cfg_dict = asdict(cfg)
    cache_path = _cache_file(cfg, model_id, tag)
    done = _read_cache(cache_path)
    keys = [_item_key(it) for it in items]
    todo = [it for it, k in zip(items, keys) if k not in done]
    if len(todo) < len(items):
        print(f"  resuming {tag}: {len(items) - len(todo)} of {len(items)} prompts already done")

    def save(batch_items, batch_results):
        with open(cache_path, "a", encoding="utf-8") as f:
            for it, r in zip(batch_items, batch_results):
                r = {**r, "_key": _item_key(it)}
                done[r["_key"]] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if todo:
        if cfg.backend in ("mock", "openai"):
            backend = load_backend(cfg.backend, model_id, cfg_dict)
            for s in range(0, len(todo), 200):
                part = todo[s:s + 200]
                save(part, backend.generate(part))
        else:
            workers = _plan_workers(model_id, cfg)
            if workers == 0:
                key = (cfg.backend, model_id)
                if key not in _BACKEND_CACHE:
                    _BACKEND_CACHE.clear()
                    print(f"Loading {model_id} in this process ...")
                    _BACKEND_CACHE[key] = load_backend(cfg.backend, model_id, cfg_dict)
                backend = _BACKEND_CACHE[key]
                big = 1024 if type(backend).__name__ == "VLLMBackend" else 128
                t0, s, n_tok, sizes = time.time(), 0, 0, []
                while s < len(todo):
                    step = 32 if s == 0 else big  # a small first chunk so the first ETA appears quickly
                    part = todo[s:s + step]
                    res = backend.generate(part, log=print)
                    save(part, res)
                    s += len(part)
                    n_tok += sum(r.get("n_out", 0) for r in res)
                    sizes += [r.get("batch_size", 1) for r in res]
                    print(progress_line(tag, s, len(todo), time.time() - t0, n_tok, sizes), flush=True)
            else:
                save(todo, _run_workers(todo, model_id, cfg_dict, workers, tag))
    out = []
    for it, k in zip(items, keys):
        r = dict(done[k])
        r.pop("_key", None)
        r["id"] = it["id"]
        out.append(r)
    return out


def release_backends() -> None:
    """Free the in-process model (call before loading a different large model)."""
    _BACKEND_CACHE.clear()
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def _run_workers(items, model_id, cfg_dict, workers, tag):
    job_dir = Path(cfg_dict["out_root"]) / "_jobs" / f"{tag}_{uuid.uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=True)
    items_path = job_dir / "items.jsonl"
    with open(items_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    job = {"backend": cfg_dict["backend"], "model_id": model_id, "items_path": str(items_path),
           "out_dir": str(job_dir), "cfg": cfg_dict}
    (job_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    if not os.path.isdir(model_id):  # download once here, not in every worker at the same time
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja",
                                                        "*.tiktoken", "*.py", "tokenizer*"])
        except Exception as e:
            print(f"  pre-download skipped ({type(e).__name__}); workers will download")
    pkg_root = str(Path(__file__).resolve().parent.parent)
    n_gpu = gpu_count()
    per = max(1, n_gpu // workers)
    procs = []
    print(f"Starting {workers} worker(s) for {len(items)} prompts with {model_id} (logs in {job_dir})")
    for k in range(workers):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in range(k * per, (k + 1) * per))
        env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
        env["TOKENIZERS_PARALLELISM"] = "false"
        log = open(job_dir / f"worker{k}.log", "w")
        p = subprocess.Popen([sys.executable, "-m", "factindic.worker", "--job", str(job_dir / "job.json"),
                              "--shard", str(k), "--nshards", str(workers)], stdout=log, stderr=subprocess.STDOUT, env=env)
        procs.append((p, log))
    last = 0.0
    while any(p.poll() is None for p, _ in procs):
        time.sleep(5)
        if time.time() - last > 60:
            last = time.time()
            for k in range(workers):
                lines = (job_dir / f"worker{k}.log").read_text(errors="ignore").strip().splitlines()
                prog = [l for l in lines if "generated" in l or "Loading" in l or "Error" in l or "OOM" in l]
                if prog:
                    print(f"  [worker {k}] {prog[-1].strip()}")
    for k, (p, log) in enumerate(procs):
        log.close()
        if p.returncode != 0:
            tail = "\n".join((job_dir / f"worker{k}.log").read_text(errors="ignore").splitlines()[-30:])
            raise RuntimeError(f"worker {k} failed (exit {p.returncode}). Last log lines:\n{tail}")
    by_id = {}
    for k in range(workers):
        with open(job_dir / f"out{k}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                by_id[r["id"]] = r
    missing = [it["id"] for it in items if it["id"] not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} results missing, e.g. {missing[:3]}")
    return [by_id[it["id"]] for it in items]
