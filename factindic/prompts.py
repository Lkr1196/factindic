"""All LLM prompts in one place.

Instructions are in English (instruction-tuned models follow English instructions most
reliably); the content and the requested output are in Hindi or Bengali.

Track A  – standard prompting: "summarise in 1–2 sentences".
Track B  – factuality-aware prompting, three mechanisms from the proposal:
           * entity locking      – names must be copied from the article, no new names;
           * number/date locks   – the numbers found in the article (extracted by code) are listed
                                   and the summary may use only those;
           * chain-of-fact       – the model first lists FACTS and ENTITIES, then writes the
                                   SUMMARY from those facts only.
Track BV – Track B followed by a verify-and-revise step: a rule-based number/date check plus
           an LLM self-check flag unsupported sentences, and the model rewrites the summary.

Every message is a list of {"role", "content"} dicts (chat format).
"""
from __future__ import annotations

import re

from .config import LANGS
from .entities import extract_dates, extract_numbers

ART_OPEN, ART_CLOSE = 'ARTICLE:\n"""\n', '\n"""'
SUM_OPEN, SUM_CLOSE = 'SUMMARY TO CHECK:\n"""\n', '\n"""'

SYSTEM_A = "You are a helpful assistant that summarizes news articles."
SYSTEM_B = ("You are a careful news editor. You write faithful summaries and never add information "
            "that the article does not state.")


def _input_note(variant: str, lang: str) -> str:
    L = LANGS[lang]
    if variant == "cm":
        return f"The article is written in code-mixed {L['name']}–English ({L['cm_name']}): {L['name']} sentences with English words in Roman letters."
    if variant == "roman":
        return f"The article is written in Romanized {L['name']} ({L['cm_name']}) with English words, all in Roman letters."
    if variant == "natural":
        return f"The article may mix {L['name']} and English."
    return f"The article is written in {L['name']}."


def article_block(text: str) -> str:
    return ART_OPEN + text.strip() + ART_CLOSE


def number_lock_list(text: str) -> str:
    nums = []
    for n in extract_numbers(text):
        if n.raw not in nums:
            nums.append(n.raw)
    dates = []
    for d, m in extract_dates(text):
        lab = f"{d}/{m}" if d else f"month {m}"
        if lab not in dates:
            dates.append(lab)
    parts = []
    if nums:
        parts.append("numbers: " + ", ".join(nums[:40]))
    if dates:
        parts.append("dates (day/month): " + ", ".join(dates[:20]))
    return "; ".join(parts) if parts else "none"


def summarize_A(text: str, lang: str, variant: str, max_words: int) -> list[dict]:
    L = LANGS[lang]
    user = (f"{_input_note(variant, lang)}\n\n{article_block(text)}\n\n"
            f"Summarize the article in {L['name']} ({L['script']} script) in 1-2 sentences "
            f"(at most {max_words} words). Output only the summary.")
    return [{"role": "system", "content": SYSTEM_A}, {"role": "user", "content": user}]


def summarize_B(text: str, lang: str, variant: str, max_words: int) -> list[dict]:
    L = LANGS[lang]
    user = f"""{_input_note(variant, lang)}

{article_block(text)}

Numbers and dates that appear in the article (extracted automatically): {number_lock_list(text)}

Write a faithful summary of this article in {L['name']} ({L['script']} script). Rules:
1. Use ONLY information stated in the article. Do not add background knowledge, causes, opinions or details that the article does not state.
2. Entity lock: write the names of people, places and organisations exactly as the article gives them (you may write them in {L['script']} script). Do not introduce any other name.
3. Number and date lock: every number, amount, percentage, year and date in your summary must be one of the values listed above, with the same meaning. Never change a year, a unit or what a number refers to.
4. The article may mix {L['name']} and English. Read English words as part of the sentence and translate them correctly; do not guess.
5. Length: 1-2 sentences, at most {max_words} words.

Work in three steps and use exactly this format:
FACTS:
- (3 or 4 key facts, one short line each, taken directly from the article)
ENTITIES: (comma-separated names, exactly as in the article)
SUMMARY: (the final summary, built only from the FACTS above)"""
    return [{"role": "system", "content": SYSTEM_B}, {"role": "user", "content": user}]


def selfcheck(text: str, summary_sents: list[str], lang: str) -> list[dict]:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(summary_sents))
    user = f"""{article_block(text)}

{SUM_OPEN}{numbered}{SUM_CLOSE}

You are a fact-checker. For each numbered summary sentence, decide whether every detail in it (names, numbers, dates, places, events, who did what) is directly supported by the article.
Answer with exactly one line per sentence, in this format:
1. SUPPORTED
2. UNSUPPORTED: <short reason in English>"""
    return [{"role": "system", "content": SYSTEM_B}, {"role": "user", "content": user}]


def revise(text: str, lang: str, variant: str, draft: str, problems: list[str], max_words: int) -> list[dict]:
    L = LANGS[lang]
    probs = "\n".join(f"- {p}" for p in problems)
    user = f"""{_input_note(variant, lang)}

{article_block(text)}

{SUM_OPEN}{draft}{SUM_CLOSE}

A fact-checker found these problems in the summary:
{probs}

Rewrite the summary in {L['name']} ({L['script']} script). Fix or remove every unsupported detail, keep the supported content, use only information from the article, and keep it to 1-2 sentences (at most {max_words} words).
Answer in this format:
SUMMARY: (the corrected summary)"""
    return [{"role": "system", "content": SYSTEM_B}, {"role": "user", "content": user}]


def question_generation(summary: str, lang: str, k: int) -> list[dict]:
    L = LANGS[lang]
    user = f"""{SUM_OPEN}{summary}{SUM_CLOSE}

Write up to {k} short factual questions in {L['name']} that are answered by this summary. Each answer must be a short span copied exactly from the summary: a name, place, organisation, number, date or short noun phrase (1-5 words). Ask about different facts.
Write one question per line in exactly this format:
Q: <question> || A: <answer>"""
    return [{"role": "system", "content": "You write reading-comprehension questions."},
            {"role": "user", "content": user}]


def codemix_rewrite(sents: list[str], lang: str) -> list[dict]:
    L = LANGS[lang]
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents))
    user = f"""Rewrite each {L['name']} sentence below as natural {L['name']}-English code-mixed text ({L['cm_name']}), the way urban Indian readers write on social media and in informal news.
Rules:
- Keep {L['name']} grammar: verb endings, postpositions and function words stay in {L['name']} ({L['script']} script).
- Replace roughly 30-50% of the content words (nouns, adjectives, common verbs, technical terms) with their natural English equivalents written in Roman (Latin) letters.
  Example: "{L['cm_example_mono']}" -> "{L['cm_example_cm']}"
- Do NOT change the meaning. Do NOT add, remove or change any fact. Every number, date and amount must stay exactly the same, written with the same digits. Keep all names of people, places and organisations.
- Rewrite every sentence. Output exactly {len(sents)} lines, numbered like the input, and nothing else.

SENTENCES:
{numbered}"""
    return [{"role": "system", "content": "You are an expert bilingual writer."},
            {"role": "user", "content": user}]


def romanize(sents: list[str], lang: str) -> list[dict]:
    L = LANGS[lang]
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents))
    user = f"""Transliterate each sentence below into Roman script, the way Indians type {L['name']} on WhatsApp (Romanized {L['name']} / {L['cm_name']}).
Rules:
- Write the {L['name']} words phonetically in Roman letters with common informal spellings.
  Example: "{L['cm_example_cm']}" -> "{L['roman_example']}"
- Keep the English words exactly as they are.
- Keep every number and date exactly as written, with the same digits. Keep all names.
- Do not translate, add or remove anything. Output exactly {len(sents)} lines, numbered like the input, and nothing else.

SENTENCES:
{numbered}"""
    return [{"role": "system", "content": "You are an expert bilingual writer."},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------------------
def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.replace("<think>", "").replace("</think>", "").strip()


def parse_summary(raw: str) -> str:
    """Extract the summary from a model answer (handles the B/BV formats and stray labels)."""
    raw = strip_think(raw)
    m = list(re.finditer(r"(?im)^\s*\**\s*(SUMMARY|सारांश|সারাংশ)\s*\**\s*[:：]\s*", raw))
    if m:
        text = raw[m[-1].end():]
        text = re.split(r"\n\s*\n|\n\s*\**\s*(?:FACTS|ENTITIES|NOTE|Note)\b", text)[0]
    else:
        text = raw
        # drop FACTS/ENTITIES blocks if the model skipped the SUMMARY label
        text = re.sub(r"(?is)FACTS:.*?(\n\s*\n|$)", "", text)
        text = re.sub(r"(?im)^\s*ENTITIES:.*$", "", text)
    text = text.strip().lstrip("*").strip().strip('"').strip()
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def parse_selfcheck(raw: str, n: int) -> dict[int, str | None]:
    """{sentence_index(1-based): None if supported else reason}."""
    raw = strip_think(raw)
    out: dict[int, str | None] = {}
    for line in raw.splitlines():
        m = re.match(r"^\s*(\d{1,2})\s*[.):]\s*(.*)$", line)
        if not m:
            continue
        k = int(m.group(1))
        if not 1 <= k <= n or k in out:
            continue
        verdict = m.group(2).strip()
        if re.match(r"(?i)\**\s*UNSUPPORTED", verdict):
            reason = verdict.split(":", 1)[1].strip() if ":" in verdict else "not supported by the article"
            out[k] = reason or "not supported by the article"
        else:
            out[k] = None
    return out


def parse_qg(raw: str) -> list[tuple[str, str]]:
    raw = strip_think(raw)
    pairs = []
    for line in raw.splitlines():
        m = re.match(r"^\s*(?:\d+[.)]\s*)?Q\s*[:：]\s*(.+?)\s*\|\|\s*A\s*[:：]\s*(.+?)\s*$", line)
        if m:
            q, a = m.group(1).strip(), m.group(2).strip().strip('"').strip()
            if q and a:
                pairs.append((q, a))
    return pairs
