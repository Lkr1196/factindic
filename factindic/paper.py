"""Build the IEEE-style research paper (Word .docx) for FactIndic.

    python -m factindic.paper --results factindic_out --out FactIndic_paper.docx
    python -m factindic.paper --out FactIndic_paper_template.docx      # no results yet

Every section is written. Numbers, tables and figures are filled in from the pipeline outputs
when they exist (``factindic_out/01_data`` ... ``07_human`` or ``07_llm_judge``); anything missing is shown as a
yellow-highlighted placeholder. Sentences that interpret the results are marked as yellow
"[Write: ...]" notes: the team writes those after looking at the numbers.

Convert to PDF with Word (File -> Save As -> PDF) or LibreOffice:
    soffice --headless --convert-to pdf FactIndic_paper.docx
"""
from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = Path(__file__).resolve().parent / "assets"
SEQ2SEQ = ["mt5", "indicbart", "mt5-lora-cm"]
LANG_NAME = {"hi": "Hindi", "bn": "Bengali"}
VAR_NAME = {"mono": "Mono", "cm": "CM", "roman": "Roman", "natural": "Natural"}

# ======================================================================================
# References (numbered in order of first citation)
# ======================================================================================
REFS = {
    "xlsum": 'T. Hasan, A. Bhattacharjee, M. S. Islam, K. Mubasshir, Y.-F. Li, Y.-B. Kang, M. S. Rahman, and R. Shahriyar, "XL-Sum: Large-scale multilingual abstractive summarization for 44 languages," in <i>Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021</i>, 2021.',
    "indicnlg": 'A. Kumar <i>et al.</i>, "IndicNLG benchmark: Multilingual datasets for diverse NLG tasks in Indic languages," in <i>Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)</i>, 2022.',
    "gluecos": 'S. Khanuja, S. Dandapat, A. Srinivasan, S. Sitaram, and M. Choudhury, "GLUECoS: An evaluation benchmark for code-switched NLP," in <i>Proc. 58th Annu. Meeting of the Association for Computational Linguistics (ACL)</i>, 2020.',
    "hingcorpus": 'R. Nayak and R. Joshi, "L3Cube-HingCorpus and HingBERT: A code mixed Hindi-English dataset and BERT language models," in <i>Proc. WILDRE-6 Workshop at LREC</i>, 2022.',
    "maynez": 'J. Maynez, S. Narayan, B. Bohnet, and R. McDonald, "On faithfulness and factuality in abstractive summarization," in <i>Proc. 58th Annu. Meeting of the Association for Computational Linguistics (ACL)</i>, 2020.',
    "rouge": 'C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," in <i>Text Summarization Branches Out (ACL Workshop)</i>, 2004.',
    "mt5": 'L. Xue <i>et al.</i>, "mT5: A massively multilingual pre-trained text-to-text transformer," in <i>Proc. Conf. North American Chapter of the Association for Computational Linguistics (NAACL-HLT)</i>, 2021.',
    "indicbart": 'R. Dabre, H. Shrotriya, A. Kunchukuttan, R. Puduppully, M. M. Khapra, and P. Kumar, "IndicBART: A pre-trained model for Indic natural language generation," in <i>Findings of the Association for Computational Linguistics: ACL 2022</i>, 2022.',
    "ilsum": 'S. Satapara, B. Modha, S. Modha, and P. Mehta, "Findings of the first shared task on Indian language summarization (ILSUM): Approaches, challenges and the path ahead," in <i>Working Notes of FIRE 2022 - Forum for Information Retrieval Evaluation</i>, CEUR-WS, 2022.',
    "factcc": 'W. Kryściński, B. McCann, C. Xiong, and R. Socher, "Evaluating the factual consistency of abstractive text summarization," in <i>Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)</i>, 2020.',
    "summac": 'P. Laban, T. Schnabel, P. N. Bennett, and M. A. Hearst, "SummaC: Re-visiting NLI-based models for inconsistency detection in summarization," <i>Transactions of the Association for Computational Linguistics</i>, vol. 10, 2022.',
    "qags": 'A. Wang, K. Cho, and M. Lewis, "Asking and answering questions to evaluate the factual consistency of summaries," in <i>Proc. 58th Annu. Meeting of the Association for Computational Linguistics (ACL)</i>, 2020.',
    "qafacteval": 'A. R. Fabbri, C.-S. Wu, W. Liu, and C. Xiong, "QAFactEval: Improved QA-based factual consistency evaluation for summarization," in <i>Proc. Conf. North American Chapter of the Association for Computational Linguistics (NAACL-HLT)</i>, 2022.',
    "nan": 'F. Nan, R. Nallapati, Z. Wang, C. Nogueira dos Santos, H. Zhu, D. Zhang, K. McKeown, and B. Xiang, "Entity-level factual consistency of abstractive text summarization," in <i>Proc. 16th Conf. European Chapter of the Association for Computational Linguistics (EACL)</i>, 2021.',
    "mface": 'R. Aharoni, S. Narayan, J. Maynez, J. Herzig, E. Clark, and M. Lapata, "Multilingual summarization with factual consistency evaluation," in <i>Findings of the Association for Computational Linguistics: ACL 2023</i>, 2023.',
    "cmi": 'A. Das and B. Gambäck, "Identifying languages at the word level in code-mixed Indian social media text," in <i>Proc. 11th Int. Conf. Natural Language Processing (ICON)</i>, 2014.',
    "hinge": 'V. Srivastava and M. Singh, "HinGE: A dataset for generation and evaluation of code-mixed Hinglish text," in <i>Proc. 2nd Workshop on Evaluation and Comparison of NLP Systems (Eval4NLP)</i>, 2021.',
    "gupshup": 'L. Mehnaz <i>et al.</i>, "GupShup: Summarizing open-domain code-switched conversations," in <i>Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)</i>, 2021.',
    "yong": 'Z.-X. Yong <i>et al.</i>, "Prompting multilingual large language models to generate code-mixed texts: The case of South East Asian languages," in <i>Proc. 6th Workshop on Computational Approaches to Linguistic Code-Switching (CALCS)</i>, 2023.',
    "entityplan": 'S. Narayan, Y. Zhao, J. Maynez, G. Simões, V. Nikolaev, and R. McDonald, "Planning with learned entity prompts for abstractive summarization," <i>Transactions of the Association for Computational Linguistics</i>, vol. 9, 2021.',
    "cove": 'S. Dhuliawala, M. Komeili, J. Xu, R. Raileanu, X. Li, A. Celikyilmaz, and J. Weston, "Chain-of-verification reduces hallucination in large language models," in <i>Findings of the Association for Computational Linguistics: ACL 2024</i>, 2024.',
    "selfrefine": 'A. Madaan <i>et al.</i>, "Self-Refine: Iterative refinement with self-feedback," in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 2023.',
    "lora": 'E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-rank adaptation of large language models," in <i>Proc. Int. Conf. Learning Representations (ICLR)</i>, 2022.',
    "qwen3": 'A. Yang <i>et al.</i> (Qwen Team), "Qwen3 technical report," arXiv preprint arXiv:2505.09388, 2025.',
    "minhash": 'A. Z. Broder, "On the resemblance and containment of documents," in <i>Proc. Compression and Complexity of Sequences</i>, 1997.',
    "debertav3": 'P. He, J. Gao, and W. Chen, "DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing," in <i>Proc. Int. Conf. Learning Representations (ICLR)</i>, 2023.',
    "laurer": 'M. Laurer, W. van Atteveldt, A. Casas, and K. Welbers, "Less annotating, more classifying: Addressing the data scarcity issue of supervised machine learning with deep transfer learning and BERT-NLI," <i>Political Analysis</i>, 2024.',
    "xlmr": 'A. Conneau <i>et al.</i>, "Unsupervised cross-lingual representation learning at scale," in <i>Proc. 58th Annu. Meeting of the Association for Computational Linguistics (ACL)</i>, 2020.',
    "squad2": 'P. Rajpurkar, R. Jia, and P. Liang, "Know what you don\'t know: Unanswerable questions for SQuAD," in <i>Proc. 56th Annu. Meeting of the Association for Computational Linguistics (ACL)</i>, 2018.',
    "naamapadam": 'A. Mhaske <i>et al.</i>, "Naamapadam: A large-scale named entity annotated data for Indic languages," in <i>Proc. 61st Annu. Meeting of the Association for Computational Linguistics (ACL)</i>, 2023.',
    "meteor": 'S. Banerjee and A. Lavie, "METEOR: An automatic metric for MT evaluation with improved correlation with human judgments," in <i>Proc. ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for MT and/or Summarization</i>, 2005.',
    "bertscore": 'T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, "BERTScore: Evaluating text generation with BERT," in <i>Proc. Int. Conf. Learning Representations (ICLR)</i>, 2020.',
    "labse": 'F. Feng, Y. Yang, D. Cer, N. Arivazhagan, and W. Wang, "Language-agnostic BERT sentence embedding," in <i>Proc. 60th Annu. Meeting of the Association for Computational Linguistics (ACL)</i>, 2022.',
    "wilcoxon": 'F. Wilcoxon, "Individual comparisons by ranking methods," <i>Biometrics Bulletin</i>, vol. 1, no. 6, pp. 80–83, 1945.',
    "mcnemar": 'Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," <i>Psychometrika</i>, vol. 12, no. 2, pp. 153–157, 1947.',
    "koehn": 'P. Koehn, "Statistical significance tests for machine translation evaluation," in <i>Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)</i>, 2004.',
    "holm": 'S. Holm, "A simple sequentially rejective multiple test procedure," <i>Scandinavian Journal of Statistics</i>, vol. 6, no. 2, pp. 65–70, 1979.',
    "cohen": 'J. Cohen, "A coefficient of agreement for nominal scales," <i>Educational and Psychological Measurement</i>, vol. 20, no. 1, pp. 37–46, 1960.',
}


# ======================================================================================
# Results
# ======================================================================================
class Results:
    """Everything the paper needs from ``factindic_out``; attributes are None when missing."""

    def __init__(self, root: str | None):
        self.root = Path(root) if root else None

        def csv(rel):
            p = self.root / rel if self.root else None
            return pd.read_csv(p, keep_default_na=True, na_values=[""]) if p and p.exists() else None

        def jsonl(rel):
            p = self.root / rel if self.root else None
            if not (p and p.exists()):
                return None
            with open(p, encoding="utf-8") as f:
                return pd.DataFrame([json.loads(l) for l in f if l.strip()])

        self.data_stats = csv("01_data/data_stats.csv")
        self.docs = jsonl("01_data/docs.jsonl")
        self.variants = jsonl("02_codemix/variants.jsonl")
        t = "06_evaluate/tables/"
        self.main = csv(t + "main_results.csv")
        self.h1 = csv(t + "h1_codemix_gap.csv")
        self.h2 = csv(t + "h2_tracks.csv")
        self.did = csv(t + "h2_gap_reduction_did.csv")
        self.cmi = csv(t + "cmi_bins.csv")
        self.fert = csv(t + "tokenizer_fertility.csv")
        self.outmix = csv(t + "output_mix.csv")
        self.errors = csv(t + "error_types.csv")
        self.natural = csv(t + "natural_results.csv")
        # Sentence labels come either from human annotators (07_human) or from an LLM judge (07_llm_judge).
        # The paper never calls LLM-judge labels human.
        jp = self.root / "07_llm_judge" / "judge.json" if self.root else None
        self.llm_judge = bool(jp and jp.exists())
        self.judge_name = json.loads(jp.read_text(encoding="utf-8")).get("name", "") if self.llm_judge else ""
        hd = "07_llm_judge/" if self.llm_judge else "07_human/"
        self.agree = None if self.llm_judge else csv(hd + "agreement.csv")
        self.detect = csv(hd + "detection.csv")
        self.hrates = csv(hd + ("judge_rates.csv" if self.llm_judge else "human_rates.csv"))
        if self.hrates is not None:
            self.hrates = self.hrates.rename(columns={"judge_halluc_rate": "human_halluc_rate"})
        self.hcorr = csv(hd + "correlation.csv")
        self.flu = csv(hd + "fluency_coverage.csv")
        self.cmq = csv(hd + "codemix_quality.csv")
        cfg = None
        for st in ("06_evaluate", "04_llm", "01_data"):
            p = self.root / st / "config.json" if self.root else None
            if p and p.exists():
                cfg = json.loads(p.read_text(encoding="utf-8"))
                break
        self.cfg = cfg
        self.env = {}
        for st in ("04_llm", "02_codemix"):
            p = self.root / st / "env.json" if self.root else None
            if p and p.exists():
                self.env[st] = json.loads(p.read_text(encoding="utf-8"))
        for df in (self.main, self.h1, self.h2, self.errors, self.outmix):
            if df is not None and "track" in df.columns:
                df["track"] = df["track"].fillna("-").astype(str)
        self.llm_system = None
        if self.main is not None:
            llm = [s for s in self.main["system"].unique() if s not in SEQ2SEQ]
            self.llm_system = llm[0] if llm else None

    @property
    def has_results(self) -> bool:
        return self.main is not None


# ======================================================================================
# Formatting helpers
# ======================================================================================
def f1(v, d=1, signed=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return f"{v:+.{d}f}".replace("-", "−") if signed else f"{v:.{d}f}"


def pretty_system(system: str, track: str = "-") -> str:
    from .analysis import pretty_system as ps

    s = ps(system)
    return f"{s} · {track}" if track not in ("-", "", None) else s


class Doc:
    """Thin layer over python-docx with IEEE conference styling."""

    def __init__(self):
        from docx import Document
        from docx.shared import Mm, Pt

        self.Pt, self.Mm = Pt, Mm
        self.doc = Document()
        self.cites: list[str] = []
        self.values: dict[str, str | None] = {}
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Mm(210), Mm(297)
        sec.top_margin, sec.bottom_margin = Mm(19), Mm(25)
        sec.left_margin = sec.right_margin = Mm(14.3)
        self.text_w = Mm(210 - 2 * 14.3)
        self.col_w = Mm((210 - 2 * 14.3 - 4.2) / 2)
        self._style_base()
        self.table_no = 0
        self.fig_no = 0
        self.eq_no = 0

    # ---------------- styles ----------------
    def _rfonts(self, rpr, latin="Times New Roman", cs="Nirmala UI"):
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn

        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rpr.insert(0, rf)
        for a in list(rf.attrib):
            if "Theme" in a or "theme" in a:
                del rf.attrib[a]
        rf.set(qn("w:ascii"), latin)
        rf.set(qn("w:hAnsi"), latin)
        rf.set(qn("w:eastAsia"), latin)
        rf.set(qn("w:cs"), cs)

    def _style_base(self):
        from docx.shared import RGBColor

        st = self.doc.styles["Normal"]
        st.font.name = "Times New Roman"
        st.font.size = self.Pt(10)
        self._rfonts(st.element.get_or_add_rPr())
        pf = st.paragraph_format
        pf.space_before = pf.space_after = self.Pt(0)
        pf.line_spacing = 1.0
        for name in ("Heading 1", "Heading 2"):
            h = self.doc.styles[name]
            h.font.name = "Times New Roman"
            h.font.size = self.Pt(10)
            h.font.bold = False
            h.font.color.rgb = RGBColor(0, 0, 0)
            self._rfonts(h.element.get_or_add_rPr())
            h.paragraph_format.keep_with_next = True
        self.doc.styles["Heading 1"].font.small_caps = True
        self.doc.styles["Heading 2"].font.italic = True

    def columns(self, n: int, first: bool = False):
        """Start a new continuous section with n text columns."""
        from docx.enum.section import WD_SECTION
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn

        sec = self.doc.sections[0] if first else self.doc.add_section(WD_SECTION.CONTINUOUS)
        sp = sec._sectPr
        cols = sp.find(qn("w:cols"))
        if cols is None:
            cols = OxmlElement("w:cols")
            sp.append(cols)
        cols.set(qn("w:num"), str(n))
        cols.set(qn("w:space"), str(int(4.2 / 25.4 * 1440)))
        return sec

    # ---------------- rich text ----------------
    TOKEN = re.compile(r"(</?i>|</?b>|</?sub>|</?sup>|</?sc>|\{c:[^}]+\}|\{v:[^}]+\}|\{todo:[^}]+\})")

    def rich(self, par, text: str, size=None, bold=False, italic=False):
        from docx.enum.text import WD_COLOR_INDEX

        state = {"i": italic, "b": bold, "sub": False, "sup": False, "sc": False}

        def run(t, highlight=False, force_italic=None):
            r = par.add_run(t)
            r.italic = state["i"] if force_italic is None else force_italic
            r.bold = state["b"]
            r.font.small_caps = state["sc"]
            if state["sub"]:
                r.font.subscript = True
            if state["sup"]:
                r.font.superscript = True
            if size:
                r.font.size = self.Pt(size)
                rpr = r._element.get_or_add_rPr()
                from docx.oxml import OxmlElement
                from docx.oxml.ns import qn

                szcs = OxmlElement("w:szCs")
                szcs.set(qn("w:val"), str(int(size * 2)))
                sz = rpr.find(qn("w:sz"))
                if sz is not None:
                    sz.addnext(szcs)
            if highlight:
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            return r

        for tok in self.TOKEN.split(text):
            if not tok:
                continue
            m = re.fullmatch(r"<(/?)(i|b|sub|sup|sc)>", tok)
            if m:
                state[m.group(2)] = not m.group(1)
            elif tok.startswith("{c:"):
                keys = [k.strip() for k in tok[3:-1].split(",")]
                nums = []
                for k in keys:
                    if k not in REFS:
                        raise KeyError(f"unknown reference {k}")
                    if k not in self.cites:
                        self.cites.append(k)
                    nums.append(self.cites.index(k) + 1)
                run(", ".join(f"[{n}]" for n in nums))
            elif tok.startswith("{v:"):
                name, _, ph = tok[3:-1].partition("|")
                val = self.values.get(name)
                if val is None:
                    run(f"[{ph or 'x.x'}]", highlight=True)
                else:
                    self.rich(par, val, size, state["b"], state["i"])
            elif tok.startswith("{todo:"):
                run(f"[Write: {tok[6:-1]}]", highlight=True, force_italic=True)
            else:
                run(tok)
        return par

    def para(self, text: str, indent=True, size=None, align="justify", space_after=0, bold=False, italic=False,
             keep_next=False):
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        pf.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "center": WD_ALIGN_PARAGRAPH.CENTER,
                        "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
        if indent:
            pf.first_line_indent = self.Mm(3.6)
        pf.space_after = self.Pt(space_after)
        pf.keep_with_next = keep_next
        self.rich(p, text, size, bold, italic)
        return p

    def item(self, label: str, text: str):
        p = self.para(f"{label}\t" + text, indent=False)
        p.paragraph_format.left_indent = self.Mm(5)
        p.paragraph_format.first_line_indent = -self.Mm(5)
        p.paragraph_format.tab_stops.add_tab_stop(self.Mm(5))
        return p

    def heading(self, num: str, title: str):
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        p = self.doc.add_paragraph(style="Heading 1")
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = self.Pt(8)
        p.paragraph_format.space_after = self.Pt(4)
        p.add_run(f"{num}. {title}" if num else title)
        return p

    def subheading(self, letter: str, title: str):
        p = self.doc.add_paragraph(style="Heading 2")
        p.paragraph_format.space_before = self.Pt(4)
        p.paragraph_format.space_after = self.Pt(2)
        p.add_run(f"{letter}. {title}")
        return p

    def equation(self, text: str):
        from docx.enum.text import WD_TAB_ALIGNMENT

        self.eq_no += 1
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = pf.space_after = self.Pt(3)
        from docx.shared import Emu

        pf.tab_stops.add_tab_stop(Emu(int(self.col_w / 2)), WD_TAB_ALIGNMENT.CENTER)
        pf.tab_stops.add_tab_stop(Emu(int(self.col_w)), WD_TAB_ALIGNMENT.RIGHT)
        self.rich(p, "\t" + text + f"\t({self.eq_no})")
        return self.eq_no

    # ---------------- tables ----------------
    def caption_table(self, title: str):
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        self.table_no += 1
        roman = _roman(self.table_no)
        p = self.doc.add_paragraph()
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = self.Pt(6)
        p.paragraph_format.keep_with_next = True
        self.rich(p, f"<sc>Table {roman}</sc>", size=8)
        p2 = self.doc.add_paragraph()
        p2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p2.paragraph_format.space_after = self.Pt(3)
        p2.paragraph_format.keep_with_next = True
        self.rich(p2, f"<sc>{title}</sc>", size=8)
        return roman

    def table(self, title: str, header: list[str], rows: list, widths_mm: list[float], size=8,
              group_rows: set | None = None, align=None, note: str | None = None):
        """rows: list of lists of cell strings (rich markup; None = placeholder)."""
        from docx.enum.table import WD_TABLE_ALIGNMENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        roman = self.caption_table(title)
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        _table_width(t, sum(widths_mm))
        align = align or (["left"] + ["center"] * (len(header) - 1))
        group_rows = group_rows or set()
        for j, (h, w) in enumerate(zip(header, widths_mm)):
            c = t.cell(0, j)
            _cell_width(c, w)
            par = c.paragraphs[0]
            par.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER if align[j] == "center" else WD_ALIGN_PARAGRAPH.LEFT
            self.rich(par, h, size=size, bold=True)
        for i, row in enumerate(rows, 1):
            if i - 1 in group_rows:
                a = t.cell(i, 0).merge(t.cell(i, len(header) - 1))
                self.rich(a.paragraphs[0], row[0], size=size, italic=True)
                _cell_shade(a, "F2F2F0")
                continue
            for j, val in enumerate(row):
                c = t.cell(i, j)
                _cell_width(c, widths_mm[j])
                par = c.paragraphs[0]
                par.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER if align[j] == "center" else WD_ALIGN_PARAGRAPH.LEFT
                if val is None:
                    self.rich(par, "{v:__missing__}", size=size)
                else:
                    self.rich(par, str(val), size=size)
        _booktabs(t)
        n_rows = len(t.rows)
        for k, r in enumerate(t.rows):
            _cant_split(r)
            for c in r.cells:
                for par in c.paragraphs:
                    par.paragraph_format.space_after = self.Pt(0)
                    par.paragraph_format.first_line_indent = 0
                    par.paragraph_format.keep_with_next = k < n_rows - 1  # keep the table on one page/column
        if note:
            p = self.para(note, indent=False, size=7)
            p.paragraph_format.space_before = self.Pt(2)
        sp = self.doc.add_paragraph()
        sp.paragraph_format.space_after = self.Pt(4)
        return roman

    # ---------------- figures ----------------
    def figure(self, path: str | None, caption: str, width, placeholder_text: str = ""):
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        self.fig_no += 1
        if path is None:
            path = _placeholder_png(placeholder_text or "Figure appears here after the run", width)
        p = self.doc.add_paragraph()
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = self.Pt(6)
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=width)
        c = self.doc.add_paragraph()
        c.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        c.paragraph_format.space_after = self.Pt(6)
        self.rich(c, f"Fig. {self.fig_no}. " + caption, size=8)
        return self.fig_no

    def save(self, path: str):
        from docx.oxml.ns import qn

        zoom = self.doc.settings.element.find(qn("w:zoom"))
        if zoom is not None and zoom.get(qn("w:percent")) is None:
            zoom.set(qn("w:percent"), "100")  # python-docx's default settings omit this required attribute
        self.doc.core_properties.title = "FactIndic: Factuality-Aware Summarization of Hindi, Bengali and Code-Mixed Text"
        self.doc.core_properties.author = "Team FactForge"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(path)


def _roman(n: int) -> str:
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


TBLPR_ORDER = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize",
               "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook",
               "tblCaption", "tblDescription"]


def _put_in_order(parent, child, order):
    """Insert ``child`` into ``parent`` respecting the OOXML schema order (Word is strict about it)."""
    from docx.oxml.ns import qn

    name = child.tag.split("}")[1]
    for old in parent.findall(qn("w:" + name)):
        parent.remove(old)
    idx = order.index(name)
    for i, existing in enumerate(list(parent)):
        ename = existing.tag.split("}")[1]
        if ename in order and order.index(ename) > idx:
            existing.addprevious(child)
            return
    parent.append(child)


def _table_width(t, mm):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tblPr = t._tbl.tblPr
    w = OxmlElement("w:tblW")
    w.set(qn("w:w"), str(int(mm / 25.4 * 1440)))
    w.set(qn("w:type"), "dxa")
    _put_in_order(tblPr, w, TBLPR_ORDER)
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    _put_in_order(tblPr, lay, TBLPR_ORDER)
    mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:w"), "40")
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    _put_in_order(tblPr, mar, TBLPR_ORDER)
    grid = t._tbl.tblGrid
    for gc in grid.findall(qn("w:gridCol")):
        grid.remove(gc)


TCPR_ORDER = ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap", "tcMar", "textDirection",
              "tcFitText", "vAlign", "hideMark"]


def _cell_width(c, mm):
    from docx.shared import Mm

    c.width = Mm(mm)
    tbl = c._tc.getparent().getparent()
    grid = tbl.tblGrid
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    idx = list(c._tc.getparent()).index(c._tc) if False else None  # noqa: F841
    # ensure gridCol entries exist (one per column)
    n_cols = len(c._tc.getparent().findall(qn("w:tc")))
    cols = grid.findall(qn("w:gridCol"))
    while len(cols) < n_cols:
        g = OxmlElement("w:gridCol")
        grid.append(g)
        cols = grid.findall(qn("w:gridCol"))
    pos = list(c._tc.getparent().findall(qn("w:tc"))).index(c._tc)
    cols[pos].set(qn("w:w"), str(int(mm / 25.4 * 1440)))


def _cant_split(row):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    trPr = row._tr.get_or_add_trPr()
    e = OxmlElement("w:cantSplit")
    trPr.append(e)


def _cell_shade(c, hexcolor):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tcPr = c._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexcolor)
    _put_in_order(tcPr, shd, TCPR_ORDER)


def _booktabs(t):
    """Top rule, rule under the header, bottom rule; no vertical lines."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    def border(cell, side, sz):
        tcPr = cell._tc.get_or_add_tcPr()
        b = tcPr.find(qn("w:tcBorders"))
        if b is None:
            b = OxmlElement("w:tcBorders")
            _put_in_order(tcPr, b, TCPR_ORDER)
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:color"), "000000")
        b.append(e)

    rows = t.rows
    for c in rows[0].cells:
        border(c, "top", 8)
        border(c, "bottom", 4)
    for c in rows[-1].cells:
        border(c, "bottom", 8)


def _placeholder_png(text: str, width) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w_in = width / 914400
    fig = plt.figure(figsize=(w_in, w_in * 0.18))
    ax = fig.add_axes([0.01, 0.03, 0.98, 0.94])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linestyle((0, (4, 3)))
        s.set_color("#898781")
    ax.set_facecolor("#fff8c5")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=7.5, color="#52514e", wrap=True,
            family="serif")
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(f.name, dpi=200)
    plt.close(fig)
    return f.name


# ======================================================================================
# Values (numbers in the text)
# ======================================================================================
def compute_values(R: Results) -> dict:
    from .config import Config

    d = Config()
    cfg = R.cfg or {}
    V: dict[str, str | None] = {}
    V["n_dev"] = str(cfg.get("n_dev", d.n_dev))
    V["qa_docs"] = str(cfg.get("qa_docs_per_lang", d.qa_docs_per_lang))
    V["human_docs"] = str(cfg.get("human_docs_per_lang", d.human_docs_per_lang))
    V["summarizer"] = _model_name(cfg.get("summarizer_model", d.summarizer_model))
    rew_cfg = (R.env.get("02_codemix") or {}).get("rewriter") or cfg.get("codemix_model", d.codemix_model)
    V["rewriter"] = _model_name(rew_cfg)
    if V["rewriter"] == V["summarizer"]:
        V["rewriter_note"] = (", the same model that later summarizes them; this keeps the experiment within free "
                              "compute, at the cost discussed in Section X")
        V["rewriter_limit"] = (" The same LLM writes the code-mixed inputs and summarizes them, so it may find "
                               "text in its own style easier to read than human code-mixing, which would make the "
                               "measured code-mixing gap a lower bound.")
    else:
        V["rewriter_note"] = ", a different model from the summarizer, so that the summarizer never reads text in its own style"
        V["rewriter_limit"] = ""
    V["n_boot"] = f"{cfg.get('n_bootstrap', d.n_bootstrap):,}"
    if R.docs is not None:
        cnt = R.docs.groupby(["lang", "role"]).size()
        V["n_test"] = _per_lang({l: cnt.get((l, "test"), 0) for l in ("hi", "bn")})
        V["n_train"] = _per_lang({l: cnt.get((l, "train"), 0) for l in ("hi", "bn")})
        if R.variants is not None and "role" in R.variants:
            tv = R.variants[R.variants["role"] == "train"]
            used = {l: tv[tv["lang"] == l]["doc_id"].nunique() for l in ("hi", "bn")}
            if sum(used.values()):
                V["n_train_used"] = _per_lang(used)
        nat = {l: cnt.get((l, "natural"), 0) for l in ("hi", "bn")}
        V["n_natural"] = _per_lang(nat) if sum(nat.values()) else "0"
    else:
        V["n_test"] = str(cfg.get("n_test", d.n_test))
        V["n_train"] = str(cfg.get("n_train_cm", d.n_train_cm))
        V["n_natural"] = None
    if R.variants is not None:
        v = R.variants
        test = v[v["role"] == "test"]
        paired = test.groupby(["lang", "doc_id"])["paired_ok"].first().groupby("lang").sum()
        V["n_paired"] = _per_lang({l: int(paired.get(l, 0)) for l in ("hi", "bn")})
        tr = v[(v["role"] == "train") & (v["valid_frac"] >= 0.8)]
        V["lora_pairs"] = f"{len(tr):,}" if len(tr) else None
    else:
        V["n_paired"] = V["lora_pairs"] = None
    # abstract / conclusion numbers: LLM hallucination rates
    if R.main is not None and R.llm_system:
        m = R.main

        def rate(lang, track, var, col="any_halluc"):
            r = m[(m["lang"] == lang) & (m["system"] == R.llm_system) & (m["track"] == track) & (m["variant"] == var)]
            return f1(r[col].iloc[0]) if len(r) and col in r else None

        for lang in ("hi", "bn"):
            for var in ("mono", "cm", "roman"):
                for tr in ("A", "B", "BV"):
                    V[f"hal_{lang}_{tr}_{var}"] = rate(lang, tr, var)
                    V[f"nli_{lang}_{tr}_{var}"] = rate(lang, tr, var, "nli_score")
    if R.cmq is not None and len(R.cmq):
        parts = []
        for r in R.cmq.itertuples(index=False):
            parts.append(f"{LANG_NAME.get(r.lang, r.lang)}: meaning kept in {r.meaning_kept:.0f}% of sentences, "
                         f"naturalness {r.cm_naturalness:.1f}/5 (code-mixed) and {r.roman_naturalness:.1f}/5 (romanised)")
        V["cm_quality"] = "; ".join(parts)
    V.update(hardware_sentences(R))
    V.update(results_sentences(R))
    return V


def hardware_sentences(R: Results) -> dict:
    """Describe where the experiments ran (defaults to the planned Kaggle setup)."""
    env = R.env.get("04_llm") or R.env.get("02_codemix") or {}
    gpus = env.get("gpus") or []
    plat = env.get("platform")
    if gpus:
        name = gpus[0].replace("Tesla ", "")
        n = len(gpus)
        mem = (env.get("gpu_mem_gb") or [16])[0]
        where = {"kaggle": "free Kaggle notebooks", "colab": "Google Colab notebooks"}.get(plat, "a workstation")
        count = {1: "one", 2: "two", 4: "four"}.get(n, str(n))
        plat_s = f"{where} with {count} NVIDIA {name} GPU{'s' if n > 1 else ''} ({mem:.0f} GB each)" if n > 1 else \
                 f"{where} with one NVIDIA {name} GPU ({mem:.0f} GB)"
    else:
        n = 2
        plat_s = "free Kaggle notebooks with two NVIDIA T4 GPUs (16 GB each)"
    q = (R.env.get("02_codemix") or {}).get("rewriter_quantization")
    from .config import Config

    rew = _model_name((R.env.get("02_codemix") or {}).get("rewriter", Config().codemix_model))
    if q:
        rew_s = (f"the {rew} rewriter is loaded in 4-bit precision (NF4) to fit in GPU memory" if q == "4bit" else
                 f"the {rew} rewriter is loaded in 8-bit precision to fit in GPU memory")
    else:
        from .llm import model_params_b

        fits_one = model_params_b(rew) * 2 * 1.25 < 0.8 * 16
        rew_s = (f"the {rew} rewriter also runs one worker per GPU" if (n >= 2 and fits_one) else
                 f"the {rew} rewriter is split across both GPUs" if n >= 2 else f"the {rew} rewriter runs on the single GPU")
    summ_s = "The LLM summarizer runs one worker per GPU, each on half of the inputs" if n >= 2 else \
             "The LLM summarizer runs on the single GPU"
    return {"platform": plat_s, "parallel": f"{summ_s}; {rew_s}."}


def _lang_sorted(df):
    order = {"hi": 0, "bn": 1}
    return df.assign(_o=df["lang"].map(order).fillna(9)).sort_values(["_o"], kind="stable").drop(columns="_o")


def _model_name(mid: str) -> str:
    return mid.split("/")[-1]


def _per_lang(d: dict) -> str:
    vals = {k: int(v) for k, v in d.items()}
    if len(set(vals.values())) == 1:
        return f"{list(vals.values())[0]:,}"
    return f"{vals.get('hi', 0):,} (Hindi) and {vals.get('bn', 0):,} (Bengali)"


def results_sentences(R: Results) -> dict:
    """Factual one-line summaries of the tests (no interpretation)."""
    V: dict[str, str | None] = {}
    if R.main is not None:
        m = R.main
        best = []
        for lang in ("hi", "bn"):
            g = m[(m["lang"] == lang) & (m["variant"] == "mono")].dropna(subset=["nli_score"])
            if len(g):
                b = g.loc[g["nli_score"].idxmax()]
                best.append(f"{pretty_system(b['system'], b['track'])} for {LANG_NAME[lang]} ({b['nli_score']:.1f})")
        V["best_mono"] = " and ".join(best) if best else None
        hal = []
        for lang in ("hi", "bn"):
            g = m[m["lang"] == lang].dropna(subset=["any_halluc"])
            if len(g):
                lo, hi = g.loc[g["any_halluc"].idxmin()], g.loc[g["any_halluc"].idxmax()]
                hal.append(f"from {lo['any_halluc']:.1f}% ({pretty_system(lo['system'], lo['track'])}, {VAR_NAME[lo['variant']]}) "
                           f"to {hi['any_halluc']:.1f}% ({pretty_system(hi['system'], hi['track'])}, {VAR_NAME[hi['variant']]}) for {LANG_NAME[lang]}")
        V["hal_range"] = "; ".join(hal) if hal else None
        eff = []
        if R.llm_system:
            for tr in ("A", "B", "BV"):
                g = m[(m["system"] == R.llm_system) & (m["track"] == tr)]
                if len(g):
                    eff.append(f"Track {tr}: {g['time_s'].mean():.2f} s per summary, {g['summary_words'].mean():.1f} words, "
                               f"compression {g['compression_ratio'].mean():.1f}×")
        V["efficiency"] = "; ".join(eff) if eff else None
    if R.h1 is not None and len(R.h1):
        out = []
        for metric, lab, worse in (("nli_score", "NLI support", "lower"), ("any_halluc", "hallucination rate", "higher")):
            for v in ("cm", "roman"):
                g = R.h1[(R.h1["metric"] == metric) & (R.h1["contrast"] == f"{v} - mono")]
                if not len(g):
                    continue
                n = len(g)
                k = int(((g["diff"] < 0) if worse == "lower" else (g["diff"] > 0)).sum())
                ks = int((((g["diff"] < 0) if worse == "lower" else (g["diff"] > 0)) & (g["p_holm"] < 0.05)).sum())
                out.append(f"{'code-mixed' if v == 'cm' else 'romanised'} input gave {worse if worse == 'lower' else 'a higher'} {lab} than monolingual input "
                           f"in {k} of {n} system–language combinations ({ks} significant after Holm correction)")
        V["h1_counts"] = "; ".join(out) if out else None
    if R.h2 is not None and len(R.h2):
        out = []
        for contrast in ("B - A", "BV - A"):
            for metric, lab, better in (("any_halluc", "hallucination rate", "lower"), ("rougeL_f", "ROUGE-L", None)):
                g = R.h2[(R.h2["metric"] == metric) & (R.h2["contrast"] == contrast)]
                if not len(g):
                    continue
                if better:
                    k = int((g["diff"] < 0).sum())
                    ks = int(((g["diff"] < 0) & (g["p_holm"] < 0.05)).sum())
                    out.append(f"{contrast.replace(' - ', ' vs. ')}: lower {lab} in {k} of {len(g)} language–input "
                               f"combinations ({ks} significant)")
                else:
                    out.append(f"mean change in {lab} {g['diff'].mean():+.1f} points (range {g['diff'].min():+.1f} to "
                               f"{g['diff'].max():+.1f})".replace("-", "−"))
        V["h2_counts"] = "; ".join(out) if out else None
    if R.did is not None and len(R.did):
        g = R.did[R.did["metric"] == "any_halluc"]
        if len(g):
            parts = [f"{LANG_NAME.get(r.lang, r.lang)} {VAR_NAME.get(r.variant, r.variant)}, Track {r.track}: "
                     f"{f1(r.did, signed=True)} points [{f1(r.ci_lo, signed=True)}, {f1(r.ci_hi, signed=True)}]"
                     for r in _lang_sorted(g).itertuples(index=False)]
            V["did"] = "; ".join(parts)
    if R.cmi is not None and len(R.cmi):
        g = R.cmi.drop_duplicates(["lang", "system", "track"]).dropna(subset=["spearman_rho"])
        parts = []
        for lang, gg in sorted(g.groupby("lang"), key=lambda x: {"hi": 0, "bn": 1}.get(x[0], 9)):
            sig = int((gg["spearman_p"] < 0.05).sum())
            parts.append(f"between {f1(gg['spearman_rho'].min(), 2, True)} and {f1(gg['spearman_rho'].max(), 2, True)} "
                         f"for {LANG_NAME.get(lang, lang)} ({sig} of {len(gg)} systems with p < 0.05)")
        V["cmi_rho"] = "; ".join(parts) if parts else None
    if R.fert is not None and len(R.fert):
        piv = R.fert.pivot_table(index=["tokenizer", "lang"], columns="variant", values="tokens_per_word")
        parts = []
        for (tok, lang), r in piv.iterrows():
            if "mono" in r and "cm" in r and not pd.isna(r.get("cm")):
                s = f"{tok.split('/')[-1]} ({LANG_NAME.get(lang, lang)}): {r['mono']:.2f} → {r['cm']:.2f}"
                if "roman" in r and not pd.isna(r.get("roman")):
                    s += f" → {r['roman']:.2f}"
                parts.append(s)
        V["fertility"] = "; ".join(parts) if parts else None
    if R.outmix is not None and len(R.outmix) and R.llm_system:
        g = R.outmix[(R.outmix["system"] == R.llm_system) & (R.outmix["variant"].isin(["cm", "roman"]))]
        if len(g):
            V["outmix"] = (f"{g['summary_latin_share'].mean():.1f}% of the letters in LLM summaries of code-mixed and "
                           f"romanised inputs were Latin, against {g['input_latin_share'].mean():.1f}% in the inputs")
    if R.errors is not None and len(R.errors):
        cols = [c for c in ("err_number", "err_date", "err_entity", "err_other") if c in R.errors]
        tot = R.errors[cols].sum()
        if tot.sum() > 0:
            share = 100 * tot / tot.sum()
            names = {"err_number": "numbers", "err_date": "dates", "err_entity": "entities", "err_other": "other content"}
            V["error_mix"] = ", ".join(f"{names[c]} {share[c]:.0f}%" for c in cols)
    if R.natural is not None and len(R.natural) and R.llm_system:
        g = R.natural[R.natural["system"] == R.llm_system]
        if len(g):
            V["natural"] = "; ".join(f"{LANG_NAME.get(r.lang, r.lang)} Track {r.track}: NLI support {r.nli_score:.1f}, "
                                     f"hallucination rate {r.any_halluc:.1f}%" for r in _lang_sorted(g).itertuples(index=False))
    if R.agree is not None and len(R.agree):
        V["kappa"] = "; ".join(f"{LANG_NAME.get(r.lang, r.lang)} κ = {f1(r.kappa_3class, 2) or 'n/a'} (three labels), "
                               f"{f1(r.kappa_binary, 2) or 'n/a'} (hallucinated or not; n = {r.n})"
                               for r in _lang_sorted(R.agree).itertuples(index=False))
    if R.detect is not None and len(R.detect):
        g = R.detect[R.detect["gold"] == "Hallucinated"]
        V["detect"] = "; ".join(f"{r.detector}: P = {f1(r.precision, 2)}, R = {f1(r.recall, 2)}, F1 = {f1(r.f1, 2)}"
                                for r in g.itertuples(index=False))
    if R.hcorr is not None and len(R.hcorr):
        names = {"nli_score": "NLI score", "qa_f1": "QA F1", "ent_prec_src": "entity precision",
                 "num_prec_src": "numeral precision"}
        V["hcorr"] = "; ".join(f"{names.get(r.metric, r.metric)} ρ = {f1(r.spearman_rho, 2, True)}"
                               for r in R.hcorr.itertuples(index=False) if not pd.isna(r.spearman_rho))
    return V


# ======================================================================================
# Tables from results
# ======================================================================================
def systems_order(R: Results) -> list[tuple[str, str]]:
    if R.main is not None:
        keys = list(dict.fromkeys(zip(R.main["system"], R.main["track"])))
        order = {s: i for i, s in enumerate(SEQ2SEQ)}
        return sorted(keys, key=lambda k: (order.get(k[0], 9), k[0], {"A": 0, "B": 1, "BV": 2}.get(k[1], 0)))
    llm = "qwen3-4b-instruct-2507"
    return [("mt5", "-"), ("indicbart", "-"), ("mt5-lora-cm", "-"), (llm, "A"), (llm, "B"), (llm, "BV")]


def main_table_rows(R: Results):
    cols = ["nli_score", "any_halluc", "qa_f1", "ent_prec_src", "num_prec_src", "rougeL_f", "bertscore_f",
            "labse_ref", "summary_words", "compression_ratio", "time_s"]
    rows, groups = [], set()
    for lang in ("hi", "bn"):
        groups.add(len(rows))
        rows.append([LANG_NAME[lang]] + [""] * (len(cols) + 1))
        for sys_, tr in systems_order(R):
            for k, var in enumerate(("mono", "cm", "roman")):
                label = pretty_system(sys_, tr) if k == 0 else ""
                cells = []
                for c in cols:
                    v = None
                    if R.main is not None:
                        r = R.main[(R.main["lang"] == lang) & (R.main["system"] == sys_) & (R.main["track"] == tr)
                                   & (R.main["variant"] == var)]
                        if len(r) and c in r:
                            x = r[c].iloc[0]
                            v = "–" if pd.isna(x) else (f"{x:.2f}" if c == "time_s" else f"{x:.1f}")
                    cells.append(v)
                rows.append([label, VAR_NAME[var]] + cells)
    return rows, groups


def h1_rows(R: Results):
    rows = []
    for lang in ("hi", "bn"):
        for sys_, tr in systems_order(R):
            cells = []
            for metric in ("nli_score", "any_halluc"):
                for v in ("cm", "roman"):
                    val = None
                    if R.h1 is not None:
                        r = R.h1[(R.h1["lang"] == lang) & (R.h1["system"] == sys_) & (R.h1["track"] == tr)
                                 & (R.h1["metric"] == metric) & (R.h1["contrast"] == f"{v} - mono")]
                        if len(r):
                            r = r.iloc[0]
                            star = "*" if r["p_holm"] < 0.05 else ""
                            val = f"{f1(r['diff'], 1, True)}{star} [{f1(r['ci_lo'], 1, True)}, {f1(r['ci_hi'], 1, True)}]"
                        else:
                            val = "–"
                    cells.append(val)
            rows.append([LANG_NAME[lang], pretty_system(sys_, tr)] + cells)
    return rows


def h2_rows(R: Results):
    rows = []
    for lang in ("hi", "bn"):
        for var in ("mono", "cm", "roman"):
            cells = []
            for metric in ("any_halluc", "nli_score", "rougeL_f"):
                for contrast in ("B - A", "BV - A"):
                    val = None
                    if R.h2 is not None:
                        r = R.h2[(R.h2["lang"] == lang) & (R.h2["variant"] == var) & (R.h2["metric"] == metric)
                                 & (R.h2["contrast"] == contrast)]
                        if len(r):
                            r = r.iloc[0]
                            val = f"{f1(r['diff'], 1, True)}{'*' if r['p_holm'] < 0.05 else ''}"
                        else:
                            val = "–"
                    cells.append(val)
            did = None
            if var == "mono":
                did = "–"
            elif R.did is not None:
                r = R.did[(R.did["lang"] == lang) & (R.did["variant"] == var) & (R.did["metric"] == "any_halluc")
                          & (R.did["track"] == "BV")]
                did = (f"{f1(r.iloc[0]['did'], 1, True)} [{f1(r.iloc[0]['ci_lo'], 1, True)}, {f1(r.iloc[0]['ci_hi'], 1, True)}]"
                       if len(r) else "–")
            rows.append([LANG_NAME[lang], VAR_NAME[var]] + cells + [did])
    return rows


def data_rows(R: Results):
    def stat(lang, key, split="test"):
        if R.data_stats is None:
            return None
        r = R.data_stats[(R.data_stats["lang"] == lang) & (R.data_stats["split"] == split)]
        if not len(r) or key not in r:
            return None
        v = r[key].iloc[0]
        return "–" if pd.isna(v) else f"{int(v):,}"

    def docs(lang, fn):
        if R.docs is None:
            return None
        return fn(R.docs[R.docs["lang"] == lang])

    def var(lang, fn):
        if R.variants is None:
            return None
        return fn(R.variants[(R.variants["lang"] == lang) & (R.variants["role"] == "test")])

    def mean_of(df, variant, col, scale=1.0, d=1):
        g = df[df["variant"] == variant]
        return f"{scale * g[col].mean():.{d}f}" if len(g) else "–"

    spec = [
        ("XL-Sum test articles", lambda l: stat(l, "raw")),
        ("… after length/script filters", lambda l: stat(l, "kept")),
        ("… near-duplicates removed", lambda l: stat(l, "near_dup_removed")),
        ("Sampled test articles", lambda l: docs(l, lambda d: f"{(d['role'] == 'test').sum():,}")),
        ("Paired test articles (all variants valid)", lambda l: var(l, lambda d: f"{int(d.groupby('doc_id')['paired_ok'].first().sum()):,}")),
        ("Training articles used for LoRA", lambda l: (R.variants is not None and "role" in R.variants and (
            lambda t: f"{t[t['lang'] == l]['doc_id'].nunique():,}" if len(t) else None)(R.variants[R.variants["role"] == "train"]))
            or docs(l, lambda d: f"{(d['role'] == 'train').sum():,}")),
        ("Natural news documents", lambda l: docs(l, lambda d: f"{(d['role'] == 'natural').sum():,}")),
        ("Mean article length (words)", lambda l: docs(l, lambda d: f"{d[d['role'] == 'test']['n_words'].mean():.0f}")),
        ("Mean reference length (words)", lambda l: docs(l, lambda d: f"{d[d['role'] == 'test']['summary_ref'].dropna().map(lambda s: len(s.split())).mean():.1f}")),
        ("Valid sentences, CM / Roman (%)", lambda l: var(l, lambda d: f"{mean_of(d, 'cm', 'valid_frac', 100, 0)} / {mean_of(d, 'roman', 'valid_frac', 100, 0)}")),
        ("CMI, Mono / CM / Roman", lambda l: var(l, lambda d: " / ".join(mean_of(d, v, "cmi") for v in ("mono", "cm", "roman")))),
        ("English tokens in CM (%)", lambda l: var(l, lambda d: mean_of(d, "cm", "en_token_share", 100))),
        ("Switch rate in CM (%)", lambda l: var(l, lambda d: mean_of(d, "cm", "switch_rate", 100))),
    ]
    return [[name, fn("hi"), fn("bn")] for name, fn in spec]


def fertility_rows(R: Results):
    toks = list(dict.fromkeys(R.fert["tokenizer"])) if R.fert is not None else [
        "Qwen3-4B-Instruct-2507", "mT5_multilingual_XLSum", "xlm-roberta-base"]
    rows = []
    for t in toks:
        for lang in ("hi", "bn"):
            cells = []
            for v in ("mono", "cm", "roman"):
                val = None
                if R.fert is not None:
                    r = R.fert[(R.fert["tokenizer"] == t) & (R.fert["lang"] == lang) & (R.fert["variant"] == v)]
                    val = f"{r['tokens_per_word'].iloc[0]:.2f}" if len(r) else "–"
                cells.append(val)
            rows.append([t.split("/")[-1] if lang == "hi" else "", LANG_NAME[lang]] + cells)
    return rows


def human_rows(R: Results):
    rows = []
    for lang in ("hi", "bn"):
        for tr in ("A", "B"):
            cells = []
            for v in ("mono", "cm", "roman"):
                val = None
                if R.hrates is not None:
                    r = R.hrates[(R.hrates["lang"] == lang) & (R.hrates["track"] == tr) & (R.hrates["variant"] == v)]
                    val = f"{r['human_halluc_rate'].iloc[0]:.0f}" if len(r) else "–"
                cells.append(val)
            flu = None
            if R.flu is not None:
                r = R.flu[(R.flu["lang"] == lang) & (R.flu["track"] == tr)]
                flu = f"{r['fluency'].mean():.1f} / {r['coverage'].mean():.1f}" if len(r) else "–"
            rows.append([LANG_NAME[lang] if tr == "A" else "", f"Track {tr}"] + cells + [flu])
    return rows


def detection_rows(R: Results):
    dets = ["NLI < threshold", "entity/number check", "NLI or entity/number"]
    rows = []
    for d in dets:
        cells = []
        for k in ("precision", "recall", "f1"):
            val = None
            if R.detect is not None:
                r = R.detect[(R.detect["gold"] == "Hallucinated") & (R.detect["detector"] == d)]
                val = f"{r[k].iloc[0]:.2f}" if len(r) and not pd.isna(r[k].iloc[0]) else "–"
            cells.append(val)
        rows.append([d] + cells)
    return rows


def make_figures(R: Results, outdir: Path) -> dict:
    """Paper-sized versions of the pipeline figures (regenerated from the result tables)."""
    from . import analysis as A

    figs = {}
    if not R.has_results:
        return figs
    outdir.mkdir(parents=True, exist_ok=True)
    old = A.FIG_SCALE
    A.FIG_SCALE = 0.66
    try:
        jobs = [("hal", A.fig_by_variant, (R.main, "any_halluc", str(outdir / "hal.png"), "")),
                ("gap", A.fig_gap, (R.h1, "nli_score", str(outdir / "gap.png"), "")),
                ("cmi", A.fig_cmi, (R.cmi, "nli_score", str(outdir / "cmi.png"), "")),
                ("err", A.fig_errors, (R.errors, str(outdir / "err.png")))]
        for key, fn, args in jobs:
            try:
                if args[0] is not None and len(args[0]):
                    fn(*args)
                    p = Path(next(a for a in args if isinstance(a, str) and a.endswith(".png")))
                    if p.exists():
                        figs[key] = str(p)
            except Exception as e:  # never fail the paper over a figure
                print(f"  figure {key} skipped: {e}")
    finally:
        A.FIG_SCALE = old
    return figs


# ======================================================================================
# The paper
# ======================================================================================
AUTHORS = ["Aryan Singh", "Lakshya Kumar", "Udayveer Singh", "Kushagra Agarwal", "Udit Malik"]
IDS = ["S24CSEU0954", "S24CSEU0089", "S24CSEU0094", "S24CSEU0059", "S24CSEU0099"]
AFFIL = "School of Computer Science Engineering and Technology, Bennett University, Greater Noida, India"


def build_paper(results_root: str | None, out_path: str, title: str | None = None) -> str:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Mm

    R = Results(results_root)
    D = Doc()
    D.values = compute_values(R)
    figs = make_figures(R, Path(tempfile.mkdtemp(prefix="factindic_paper_")))
    full_w = D.text_w

    # ---------------- title block (one column) ----------------
    D.columns(1, first=True)
    p = D.para(title or "FactIndic: Factuality-Aware Summarization of Hindi, Bengali and Code-Mixed Text",
               indent=False, size=22, align="center", space_after=10)
    p = D.para(", ".join(AUTHORS), indent=False, size=11, align="center", space_after=2)
    D.para(f"<i>{AFFIL}</i>", indent=False, size=10, align="center", space_after=1)
    D.para(" · ".join(IDS), indent=False, size=8, align="center", space_after=12)

    # ---------------- two columns ----------------
    D.columns(2)
    D.para("<b><i>Abstract</i>—Large language models write fluent summaries, but fluency does not guarantee that a "
           "summary is faithful to its source, and the problem is little studied for Indian languages and for the "
           "code-mixed text in which many Indians write. We present FactIndic, a study of factuality-aware "
           "summarization for Hindi and Bengali. We build a paired benchmark in which every XL-Sum test article "
           "appears in three versions with identical facts: the original monolingual text, a code-mixed rewrite with "
           "English content words in Latin script, and a fully romanised rewrite, each sentence validated "
           "automatically. We compare fine-tuned mT5 and IndicBART summarizers, a LoRA-adapted mT5 and an "
           "instruction-tuned LLM under standard prompting, factuality-aware prompting (entity locking, number and "
           "date constraints, chain-of-fact) and a verify-and-revise variant, and verify every summary against the "
           "original article with multilingual NLI, question answering and cross-script entity, numeral and date "
           "alignment, " + ("compared with the sentence labels of an LLM judge" if R.llm_judge else "validated by human judgements") + ". With standard prompting, the share of LLM summaries with a "
           "detected hallucination was {v:hal_hi_A_mono}% for monolingual Hindi input, {v:hal_hi_A_cm}% for "
           "code-mixed and {v:hal_hi_A_roman}% for romanised input ({v:hal_bn_A_mono}%, {v:hal_bn_A_cm}% and "
           "{v:hal_bn_A_roman}% for Bengali); factuality-aware prompting with verification gave {v:hal_hi_BV_cm}% "
           "on code-mixed Hindi and {v:hal_bn_BV_cm}% on code-mixed Bengali. Code, prompts and the annotation "
           "protocol are released for reproducible evaluation.</b>", indent=False, size=9, space_after=4)
    D.para("<b><i>Index Terms</i>—abstractive summarization, factual consistency, hallucination, code-mixing, "
           "Hindi, Bengali, large language models, natural language inference.</b>", indent=False, size=9,
           space_after=4)

    # ---------------- I. Introduction ----------------
    D.heading("I", "Introduction")
    D.para("Automatic summarization condenses a document into a short text that keeps its important information. "
           "Digital content in Indian languages—news, government notices, social media and blogs—has grown "
           "rapidly, and summarization is an obvious way to make it manageable. Summarization for Indian languages "
           "nevertheless remains hard: the languages are morphologically rich, are written in several scripts, and "
           "have far fewer annotated resources than English {c:xlsum,indicnlg}.", indent=False)
    D.para("Many Indian readers and writers also mix languages. In code-mixed text an Indian language and English "
           "alternate inside the same sentence, as in “भारत में pollution का level लगातार increase हो रहा है” "
           "(“the pollution level in India is rising continuously”), and the mixed text is often typed in Roman "
           "script. The switch itself carries register and emphasis; it is not a translation error. For language "
           "models it is still difficult: subword tokenizers built for monolingual text fragment mixed words, "
           "code-mixed text is rare in pre-training data, and models tend to normalise the mix into one language "
           "{c:gluecos,hingcorpus}.")
    D.para("Modern summarizers and large language models (LLMs) produce fluent output, but fluency is not "
           "faithfulness. A summary can change an entity, a number or a date, or add information that the source "
           "never states {c:maynez}. Overlap metrics such as ROUGE {c:rouge} reward such a summary as long as it "
           "shares words with the reference: if a source reports inflation of 6.2 percent in 2023 and the summary "
           "says 2024, the summary is fluent, scores well and is wrong. Factual consistency has to be measured "
           "separately.")
    D.para("This paper asks whether concise and fluent summaries of Indic and code-mixed text can keep the facts "
           "and meaning of the original. We test two falsifiable hypotheses. <b>H1:</b> code-mixed input measurably "
           "degrades the factual consistency of summaries compared with monolingual input carrying the same facts. "
           "<b>H2:</b> factuality-aware prompting reduces hallucination without degrading summary quality, and "
           "narrows the gap identified in H1. We study Hindi, the highest-resource Indian language with the most "
           "code-mixed data (Hinglish), and Bengali, which tests whether the findings generalise beyond Hindi. "
           "Our contributions are:")
    D.item("1)", "a paired evaluation protocol in which XL-Sum {c:xlsum} test articles are rewritten into "
                 "code-mixed and romanised versions with identical, automatically validated facts, separating the "
                 "effect of lexical mixing (mono → code-mixed) from the effect of script (code-mixed → romanised);")
    D.item("2)", "a controlled comparison of three summarizer families—fine-tuned mT5 and IndicBART, a LoRA-adapted "
                 "mT5, and an LLM—and three prompting strategies: standard, factuality-aware, and verify-and-revise;")
    D.item("3)", "a multilingual factuality verification layer (NLI entailment, question answering, and cross-script "
                 "entity, numeral and date alignment) applied against the original monolingual article, so that "
                 "verifier quality does not vary with the input, and "
                 + ("compared with the labels of an LLM judge;" if R.llm_judge else "validated against human labels;"))
    D.item("4)", "an analysis of how code-mixing affects summarization through the Code-Mixing Index, tokenizer "
                 "fertility, the script of the output and an error typology, with paired significance tests.")

    # ---------------- II. Related work ----------------
    D.heading("II", "Related Work")
    D.subheading("A", "Summarization for Indian Languages")
    D.para("XL-Sum {c:xlsum} provides about one million BBC article–summary pairs in 44 languages, including Hindi "
           "and Bengali, and fine-tunes mT5 {c:mt5} on all of them. IndicBART {c:indicbart} is a compact "
           "sequence-to-sequence model pre-trained on Indian languages and English; it and the IndicNLG benchmark "
           "{c:indicnlg} cover summarization among other generation tasks. The ILSUM shared task {c:ilsum} targeted "
           "summarization of Indian-language news and noted code-mixing and script mixing in real news text. These "
           "works report overlap metrics; they do not measure factual consistency or compare monolingual with "
           "code-mixed input.", indent=False)
    D.subheading("B", "Factual Consistency of Summaries")
    D.para("Maynez et al. {c:maynez} showed that abstractive summarizers frequently produce content unsupported by "
           "the source. Three families of metrics followed. Entailment-based methods test whether the source "
           "entails the summary: FactCC {c:factcc} trains a classifier on synthetic errors, and SummaC {c:summac} "
           "applies off-the-shelf NLI models at sentence level. QA-based methods, QAGS {c:qags} and QAFactEval "
           "{c:qafacteval}, generate questions from the summary and compare the answers found in the summary and in "
           "the source. Entity-level metrics {c:nan} count summary entities that do not appear in the source. Most "
           "of this work is English-only; Aharoni et al. {c:mface} extended NLI-based factuality evaluation to the "
           "XL-Sum languages. We adapt all three families to Hindi, Bengali and code-mixed text.", indent=False)
    D.subheading("C", "Code-Mixing")
    D.para("The Code-Mixing Index (CMI) {c:cmi} quantifies how mixed a text is from word-level language tags. "
           "Code-mixed Hindi–English resources include GLUECoS {c:gluecos}, L3Cube-HingCorpus {c:hingcorpus} and "
           "HinGE {c:hinge}; GupShup {c:gupshup} summarizes Hinglish conversations but has no monolingual parallel, "
           "so it cannot isolate the effect of mixing. Yong et al. {c:yong} showed that multilingual LLMs can be "
           "prompted to write natural code-mixed text, which motivates our use of an LLM to build controlled "
           "code-mixed inputs.", indent=False)
    D.subheading("D", "Reducing Hallucination")
    D.para("Planning with entity chains {c:entityplan} reduces entity hallucination in fine-tuned summarizers. For "
           "LLMs, Chain-of-Verification {c:cove} and Self-Refine {c:selfrefine} ask the model to check and revise its "
           "own output. Our Track B adapts entity planning and explicit numeric constraints to prompting, and Track "
           "BV adds a verify-and-revise step. Low-rank adaptation (LoRA) {c:lora} adapts a fine-tuned summarizer to "
           "a new input distribution cheaply.", indent=False)

    D.columns(1)
    D.figure(str(ASSETS / "pipeline.png"), "Overview of FactIndic. Each test article is summarized in three versions "
             "with identical facts; every summary is verified against the original monolingual article.",
             width=full_w)
    D.columns(2)

    # ---------------- III. Problem ----------------
    D.heading("III", "Problem Formulation")
    D.para("Given an input document <i>D</i>, a summarizer <i>f</i><sub>θ</sub> produces a summary", indent=False)
    D.equation("<i>S</i> = <i>f</i><sub>θ</sub>(<i>D</i>)")
    D.para("where <i>D</i> may be monolingual Hindi or Bengali, code-mixed with English, or romanised. <i>S</i> must "
           "satisfy three objectives: (1) <i>content preservation</i>—it keeps the important information of <i>D</i>; "
           "(2) <i>meaning preservation</i>—it represents the meaning of <i>D</i> correctly, including that of "
           "code-mixed expressions; and (3) <i>factual consistency</i>—every claim in <i>S</i> is supported by "
           "<i>D</i>. Let <i>D</i><sub>m</sub> be a monolingual article and <i>D</i><sub>c</sub>, <i>D</i><sub>r</sub> "
           "its code-mixed and romanised versions with the same facts. H1 states that the factual consistency of "
           "<i>f</i><sub>θ</sub>(<i>D</i><sub>c</sub>) and <i>f</i><sub>θ</sub>(<i>D</i><sub>r</sub>), measured "
           "against <i>D</i><sub>m</sub>, is lower than that of <i>f</i><sub>θ</sub>(<i>D</i><sub>m</sub>). H2 states "
           "that a factuality-aware strategy raises consistency relative to standard prompting, does not lower "
           "content or meaning scores, and reduces the gap between <i>D</i><sub>c</sub> (or <i>D</i><sub>r</sub>) "
           "and <i>D</i><sub>m</sub>.", indent=False)

    # ---------------- IV. Data ----------------
    D.heading("IV", "Data")
    D.subheading("A", "Monolingual Backbone")
    D.para("We use the Hindi and Bengali portions of XL-Sum {c:xlsum} and keep its official training, validation "
           "and test splits: the public mT5 and IndicBART checkpoints we evaluate were fine-tuned on the official "
           "training split, so re-splitting would place their training articles in our test set. Articles are "
           "normalised (Unicode NFC, unified digits and quotation marks), BBC boilerplate such as captions and app "
           "advertisements is removed, and an article is kept if it has 120–1,200 words, a reference summary of 8–60 "
           "words, at least three sentences and at least 70% of its letters in the native script. Each article is "
           "cut to its first ~300 words at a sentence boundary (at most 14 sentences). Exact duplicates are removed, "
           "as is any test article whose 5-gram Jaccard similarity to a training or validation article, estimated "
           "with MinHash {c:minhash}, is 0.8 or more. We sample {v:n_test} test articles per language, {v:n_dev} "
           "development articles and {v:n_train} training articles" + (", of which {v:n_train_used} per language are "
           "rewritten into all three variants for adaptation (Table I)." if D.values.get("n_train_used") else
           " for adaptation (Table I)."), indent=False)
    D.subheading("B", "Paired Code-Mixed Variants")
    D.para("For every test and training article we create two variants with the same facts (Table II). The "
           "<i>code-mixed</i> variant (CM) rewrites each sentence so that grammar, verb endings, postpositions and "
           "function words stay in Hindi or Bengali in the native script, while roughly 30–50% of the content words "
           "are replaced by natural English equivalents in Latin script. The <i>romanised</i> variant (Roman) "
           "transliterates the code-mixed text fully into the informal Roman spelling used on messaging apps, "
           "keeping the English words. Mono → CM therefore changes only lexical mixing, and CM → Roman changes only "
           "the script. The rewrites are written by {v:rewriter} {c:qwen3}{v:rewriter_note}.", indent=False)
    D.para("Sentences are rewritten in numbered chunks of up to six, which keeps sentence alignment and lets every "
           "output sentence be checked against its source. A rewritten sentence must (a) contain exactly the same "
           "multiset of numbers, with Indian and English scale words resolved (5 लाख = 500,000); (b) be 0.5–1.8 "
           "times as long as the source; and (c) for CM, keep at least 25% of its letters in the native script, with "
           "at most 75% English tokens and, for sentences of six or more words, at least 8%; for Roman, contain at "
           "most 5% native-script letters. A failing sentence is retried once; if it fails again it falls back to "
           "the original sentence (CM) or a rule-based transliteration (Roman) and counts as invalid. Articles with "
           "fewer than 80% valid sentences in either variant are excluded from the paired comparison, leaving "
           "{v:n_paired} paired test articles per language. " + ("An LLM judge" if R.llm_judge else "A bilingual annotator")
           + " rates 20 sentence triples per "
           "language for meaning preservation and naturalness ({v:cm_quality|check of the rewrites}).")
    D.subheading("C", "Measuring Code-Mixing")
    D.para("Each token is tagged as the matrix language (any Devanagari or Bengali-script word); as English (a "
           "Latin-script word in native-script text or, in romanised text, a Latin word that is frequent in English "
           "and not a common romanised Hindi or Bengali word); or as language-independent (numbers, punctuation). "
           "Following {c:cmi},", indent=False)
    D.equation("CMI = 100 · [1 − max<sub><i>i</i></sub> <i>w</i><sub><i>i</i></sub> / (<i>n</i> − <i>u</i>)]")
    D.para("where <i>w</i><sub><i>i</i></sub> is the number of tokens of language <i>i</i>, <i>n</i> the number of "
           "tokens and <i>u</i> the number of language-independent tokens (CMI = 0 if <i>n</i> = <i>u</i>). We also "
           "report the share of English tokens and the switch rate, the share of adjacent language-tagged tokens "
           "that change language.", indent=False)
    D.subheading("D", "Naturally Code-Mixed Documents")
    D.para("As an ecological check we collect recent Hindi and Bengali news from public RSS feeds and documents "
           "gathered by the team ({v:n_natural} per language), split 70/10/20 at article level. They have no "
           "reference summaries and are evaluated reference-free, for factual consistency only.", indent=False)

    D.table("Dataset statistics (test split unless noted)", ["", "Hindi", "Bengali"], data_rows(R),
            [50, 18, 18], size=8)
    D.table("Example of the three paired inputs",
            ["Input", "Example"],
            [["Mono", "भारत में प्रदूषण का स्तर लगातार बढ़ रहा है।"],
             ["CM", "भारत में pollution का level लगातार increase हो रहा है।"],
             ["Roman", "Bharat mein pollution ka level lagatar increase ho raha hai."],
             ["Mono", "শহরে বায়ু দূষণের মাত্রা দ্রুত বাড়ছে।"],
             ["CM", "শহরে air pollution-এর level দ্রুত increase হচ্ছে।"],
             ["Roman", "Shohore air pollution-er level druto increase hocche."]],
            [12, 74], size=8, align=["left", "left"],
            note="Top: Hindi, “the pollution level in India is rising continuously”. Bottom: Bengali, “the level of "
                 "air pollution in the city is rising fast”. Prompt examples; real rewrites are generated per article.")

    # ---------------- V. Summarization systems ----------------
    D.heading("V", "Summarization Systems")
    D.subheading("A", "Fine-Tuned Summarizers")
    D.para("<i>mT5-XLSum</i> is mT5-base fine-tuned on all XL-Sum languages {c:xlsum,mt5}; we decode with beam search "
           "(4 beams), at most 84 tokens and no repeated bigrams, from inputs truncated to 512 tokens. "
           "<i>IndicBART-XLSum</i> {c:indicbart} is IndicBART fine-tuned on the Indian languages of XL-Sum; it reads "
           "all Indian languages in Devanagari, so Bengali input is mapped to Devanagari by Unicode block offset and "
           "the output mapped back. Both run in 32-bit precision, because T5-family models overflow in half "
           "precision. To test whether brief exposure to code-mixed input closes the gap, we train a LoRA adapter "
           "{c:lora} (rank 16, α = 32, dropout 0.05, on the attention query, key, value and output projections) on "
           "mT5-XLSum using the training articles in all three variants with their reference summaries "
           "({v:lora_pairs} pairs; 2 epochs, learning rate 3 × 10<sup>−4</sup>, effective batch size 16, 10% of the "
           "articles held out). We call it <i>mT5 + LoRA (CM)</i>.", indent=False)
    D.subheading("B", "LLM Prompting Tracks")
    D.para("The LLM summarizer is {v:summarizer} {c:qwen3} in half precision with greedy decoding and no repetition "
           "penalty, since a repetition penalty would also penalise copying names and numbers from the article. "
           "Every track asks for one or two sentences of at most 40 words in the native script of the article's "
           "language, and every prompt states whether the input is monolingual, code-mixed or romanised. Inputs, "
           "model, length budget and decoding are identical across tracks; only the prompting differs.", indent=False)
    D.para("<i>Track A (standard prompting)</i> asks only for the summary. <i>Track B (factuality-aware "
           "prompting)</i> combines three mechanisms. <i>Entity locking</i> requires the names of people, places and "
           "organisations to be written exactly as the article gives them and forbids new names. <i>Number and date "
           "constraints</i> list the numbers and dates found in the article—extracted by code from all three digit "
           "systems—and require every number, year and date in the summary to be one of them, with the same "
           "meaning. <i>Chain-of-fact prompting</i> asks the model first to list three to five FACTS and the "
           "ENTITIES and then to write the SUMMARY only from those facts; the summary is parsed from the final "
           "section. The prompt also tells the model to read English words in code-mixed input as part of the "
           "sentence and to translate them correctly.")
    D.para("<i>Track BV (verify and revise)</i> checks each Track B summary twice: a rule-based check flags numbers "
           "and dates absent from the article, and the LLM states, sentence by sentence, whether every detail is "
           "supported by the article. Flagged summaries are rewritten once together with the list of problems, in "
           "the spirit of Chain-of-Verification {c:cove} and Self-Refine {c:selfrefine}. The verifiers of Section VI "
           "are not used here, so the evaluation stays independent of the intervention.")

    # ---------------- VI. Verification ----------------
    D.heading("VI", "Factuality Verification")
    D.para("Every summary is verified against the original monolingual article <i>D</i><sub>m</sub>. Because "
           "<i>D</i><sub>m</sub>, <i>D</i><sub>c</sub> and <i>D</i><sub>r</sub> carry the same facts, this measures "
           "the summarizer rather than how well the verifiers read code-mixed text. Scores against the actual input "
           "are reported as a secondary measure; for natural documents only those are possible.", indent=False)
    D.subheading("A", "NLI Entailment")
    D.para("We use mDeBERTa-v3-base {c:debertav3} fine-tuned for NLI on 27 languages including Hindi and Bengali "
           "{c:laurer}. Following SummaC {c:summac}, each summary sentence is paired with every source sentence and "
           "every window of three consecutive source sentences, and its <i>support</i> is the maximum entailment "
           "probability over these premises. A sentence is unsupported if its support is below 0.5. A summary's NLI "
           "score is the mean support of its sentences; it is flagged if any sentence is unsupported.", indent=False)
    D.subheading("B", "QA-Based Consistency")
    D.para("QAGS {c:qags} and QAFactEval {c:qafacteval} are English-only, so we build a multilingual analogue. The "
           "LLM writes up to five factoid questions per summary whose answers are spans of the summary (names, "
           "numbers, dates, short noun phrases). An extractive QA model, XLM-RoBERTa-large {c:xlmr} fine-tuned on "
           "SQuAD 2.0 {c:squad2}, answers each question from the article (384-token windows, stride 128) and may "
           "return no answer. The QA score is the mean token F1 between the two answers, with numbers compared by "
           "value and names matched across scripts; unanswerable questions score 0. QA is computed on {v:qa_docs} "
           "test articles per language.", indent=False)
    D.subheading("C", "Entity, Numeral and Date Alignment")
    D.para("Named entities in summaries and references are tagged with IndicNER {c:naamapadam}, supplemented by "
           "capitalised Latin-script spans. Numbers are extracted in all three digit systems and resolved to values "
           "with Indian and English scale words (lakh, crore, million); dates are (day, month) pairs with month "
           "names in English, Hindi and Bengali. The same name can appear in different scripts (दिल्ली / Delhi) or "
           "with Bengali case endings (কলকাতায়, “in Kolkata”), so every word is mapped to a loose consonant "
           "skeleton of its transliteration, and skeletons are matched allowing a short suffix. Following Nan et "
           "al. {c:nan}, <i>precision against the source</i> is the share of summary entities (numbers, dates) found "
           "in the article—its complement measures hallucination—and precision, recall and F1 against the "
           "reference measure content preservation. A summary has an <i>entity/number/date hallucination</i> if any "
           "of these is unsupported, and <i>any hallucination</i> if it has one or an unsupported NLI sentence. Each "
           "unsupported sentence gets an error type, in the order number, date, entity, other.", indent=False)

    # ---------------- VII. Setup ----------------
    D.heading("VII", "Experimental Setup")
    D.subheading("A", "Metrics")
    D.para("ROUGE-1/2/L {c:rouge} use a tokenizer that splits on whitespace and punctuation (including the danda) and "
           "keeps Indic vowel signs attached; the common English ROUGE tokenizer deletes all non-Latin characters. "
           "Our ROUGE has no stemming and is lower than published XL-Sum numbers. METEOR {c:meteor} uses only its "
           "exact-match module, since its stem and synonym modules are English-only. BERTScore {c:bertscore} uses "
           "multilingual BERT (layer 9), and LaBSE {c:labse} cosine similarity measures script-independent semantic "
           "similarity to the reference. Efficiency is measured by average summary length in words, compression "
           "ratio (input words ÷ summary words) and inference time per summary. Table III maps the metrics to the "
           "three objectives.", indent=False)
    D.table("Metrics for each objective", ["Objective", "Metrics"],
            [["Content", "ROUGE-1/2/L (P/R/F1), METEOR, entity and numeral P/R/F1 against the reference"],
             ["Meaning", "BERTScore (P/R/F1), LaBSE similarity to the reference"],
             ["Factual consistency", "NLI support and hallucination rate, QA F1, entity/numeral/date precision "
                                     "against the source, any-hallucination rate"],
             ["Efficiency", "Average summary length, compression ratio, inference time"]],
            [22, 64], size=8, align=["left", "left"])
    D.subheading("B", "Statistical Testing")
    D.para("All comparisons are paired over the same articles and use only articles valid in every variant. We use "
           "the Wilcoxon signed-rank test {c:wilcoxon} for scores and the exact McNemar test {c:mcnemar} for binary "
           "hallucination flags, report mean differences with 95% bootstrap confidence intervals ({v:n_boot} "
           "resamples) {c:koehn}, and correct p-values within each family of tests with Holm's method {c:holm}. "
           "Whether factuality-aware prompting narrows the code-mixing gap is tested per article with the "
           "difference-in-differences (CM<sub>B</sub> − Mono<sub>B</sub>) − (CM<sub>A</sub> − Mono<sub>A</sub>).",
           indent=False)
    if R.llm_judge:
        D.subheading("C", "LLM-Judge Evaluation")
        D.para("No annotator fluent in Bengali was available, so the sentence-level check that the protocol assigns "
               "to human annotators was carried out by an LLM judge" + (f" ({R.judge_name})" if R.judge_name else "")
               + ", a different model from the summarizer. For {v:human_docs} test articles per language the judge "
               "sees the original article and six shuffled summaries without system labels (Tracks A and B on the "
               "three inputs). It labels every summary sentence Supported, Partial or Hallucinated, gives an error "
               "type (entity, number, date, event or relation, extrinsic information, misread code-mixing, other), "
               "and rates each summary's fluency and coverage from 1 to 5. Each sentence is labelled once, so "
               "inter-annotator agreement {c:cohen} is not available. The labels are used to estimate the precision, "
               "recall and F1 of the automatic checkers and to look at H1 and H2 from a second, independent "
               "reading; they are not human judgements and are not reported as such.", indent=False)
    else:
        D.subheading("C", "Human Evaluation")
        D.para("For {v:human_docs} test articles per language, annotators fluent in the language see the original "
               "article and six shuffled summaries without system labels (Tracks A and B on the three inputs). They "
               "label every summary sentence Supported, Partial or Hallucinated, give an error type (entity, number, "
               "date, event or relation, extrinsic information, misread code-mixing, other), and rate each summary's "
               "fluency and coverage from 1 to 5. A quarter of the articles are labelled by two annotators to compute "
               "Cohen's κ {c:cohen} (target above 0.6). The labels validate the automatic checkers (precision, recall "
               "and F1 for detecting hallucinated sentences) and test H1 and H2 by hand.", indent=False)
    D.subheading("D", "Implementation")
    D.para("All experiments run on {v:platform} using PyTorch and Hugging Face Transformers. {v:parallel} "
           "Inference times are per GPU under batched decoding (batch time divided by batch size). The full "
           "pipeline used {v:gpu_hours|x} GPU hours.", indent=False)

    # ---------------- VIII. Results ----------------
    D.heading("VIII", "Results")
    D.subheading("A", "Overall Results")
    D.para("Table IV reports every system on the paired test articles. On monolingual input the highest NLI support "
           "came from {v:best_mono|best system per language}. Across all systems and inputs the share of summaries "
           "with any detected hallucination ranged {v:hal_range|lowest to highest hallucination rate}. "
           "{todo:2–3 sentences on how the fine-tuned models, the LoRA model and the LLM tracks compare on factuality "
           "versus ROUGE/BERTScore, and whether the two languages behave alike.}", indent=False)
    D.columns(1)
    mrows, mgroups = main_table_rows(R)
    D.table("Results on the paired test articles (rates and 0–1 scores in %)",
            ["System", "Input", "NLI↑", "Hal.↓", "QA↑", "Ent-P↑", "Num-P↑", "R-L↑", "BS↑", "LaBSE↑", "Len", "CR",
             "Time(s)"],
            mrows, [34, 12, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 12], size=7, group_rows=mgroups,
            note="NLI: mean NLI support; Hal.: summaries with any detected hallucination; QA: QA consistency F1; "
                 "Ent-P/Num-P: entity/numeral precision against the source; R-L: ROUGE-L F1; BS: BERTScore F1; "
                 "LaBSE: similarity to the reference; Len: words; CR: compression ratio; Time: seconds per summary. "
                 "Mono/CM/Roman: monolingual, code-mixed, romanised input.")
    D.columns(2)
    D.subheading("B", "H1: Effect of Code-Mixed Input")
    D.para("Table V gives the paired change from monolingual to code-mixed and romanised input. In summary, "
           "{v:h1_counts|counts of systems with lower factuality on code-mixed and romanised input}. Fig. 2 shows "
           "the paired change in NLI support and Fig. 3 the hallucination rates by input. {todo:State whether H1 is "
           "supported, for which systems and languages, and whether mixing (Mono → CM) or script (CM → Roman) costs "
           "more.}", indent=False)
    D.columns(1)
    D.table("H1: paired change from monolingual input (percentage points, 95% CI)",
            ["Lang", "System", "ΔNLI CM", "ΔNLI Roman", "ΔHal. CM", "ΔHal. Roman"], h1_rows(R),
            [16, 34, 32, 32, 32, 32], size=7,
            note="* significant after Holm correction (p < 0.05). Wilcoxon signed-rank test for NLI support, exact "
                 "McNemar test for hallucination flags.")
    D.figure(figs.get("gap"), "Paired change in NLI support from monolingual to code-mixed and romanised input "
             "(95% bootstrap CIs).", width=full_w,
             placeholder_text="Fig. 2 appears here after the run (code-mixing gap in NLI support)")
    D.figure(figs.get("hal"), "Share of summaries with any detected hallucination, by system and input type "
             "(95% bootstrap CIs).", width=full_w,
             placeholder_text="Fig. 3 appears here after the run (hallucination rate by input type)")
    D.columns(2)
    D.subheading("C", "H2: Factuality-Aware Prompting")
    D.para("Table VI compares the LLM tracks. {v:h2_counts|counts of language–input combinations where Track B/BV "
           "lowered the hallucination rate, and the change in ROUGE-L}. The difference-in-differences for the "
           "hallucination rate was {v:did|DiD per language and input}. {todo:State whether H2 is "
           "supported: does Track B or BV reduce hallucination, at what cost in ROUGE/BERTScore and inference time, "
           "and does it shrink the code-mixing gap?}", indent=False)
    D.columns(1)
    D.table("H2: change relative to Track A (percentage points) and gap reduction",
            ["Lang", "Input", "ΔHal. B", "ΔHal. BV", "ΔNLI B", "ΔNLI BV", "ΔR-L B", "ΔR-L BV", "DiD Hal. BV"],
            h2_rows(R), [16, 14, 17, 17, 17, 17, 17, 17, 46], size=7,
            note="Δ = Track B (or BV) − Track A on the same articles; * Holm-corrected p < 0.05. DiD: (CM or Roman − "
                 "Mono) under BV minus the same gap under A, with 95% CI; negative values mean BV narrows the gap.")
    D.columns(2)
    D.subheading("D", "Code-Mixing Analysis")
    D.para("Table VII lists tokenizer fertility. From monolingual to code-mixed to romanised input, subword tokens "
           "per word went {v:fertility|tokens per word, mono → CM → Roman}. The correlation between the input's CMI "
           "and NLI support (Fig. 4), Spearman's ρ, was {v:cmi_rho|Spearman ρ per language}. In the output, {v:outmix|share of "
           "Latin-script letters in summaries vs. inputs}. Among unsupported summary sentences the error types were "
           "{v:error_mix|share of number, date, entity and other errors} (Fig. 5). {todo:Explain what these "
           "patterns suggest: fragmentation, normalisation of the mix, and which error types code-mixing "
           "increases.}", indent=False)
    D.table("Tokenizer fertility (subword tokens per word)", ["Tokenizer", "Lang", "Mono", "CM", "Roman"],
            fertility_rows(R), [38, 15, 11, 11, 11], size=8)
    D.columns(1)
    D.figure(figs.get("cmi"), "NLI support by quantile bin of the input's Code-Mixing Index (code-mixed, romanised "
             "and natural inputs).", width=full_w,
             placeholder_text="Fig. 4 appears here after the run (NLI support vs. Code-Mixing Index)")
    D.figure(figs.get("err"), "Unsupported summary sentences by error type, per 100 summaries.", width=full_w,
             placeholder_text="Fig. 5 appears here after the run (error types by system and input)")
    D.columns(2)
    D.subheading("E", "Efficiency and Natural Documents")
    D.para("For the LLM, {v:efficiency|time per summary, length and compression ratio for Tracks A, B and BV}. On "
           "the natural documents collected from news feeds, {v:natural|NLI support and hallucination rate on natural documents}. "
           "{todo:Comment on the time cost of Tracks B and BV and on whether the natural documents agree with the "
           "synthetic results.}", indent=False)
    who = "LLM-judge" if R.llm_judge else "human"
    D.subheading("F", "LLM-Judge Evaluation" if R.llm_judge else "Human Evaluation")
    D.para(("" if R.llm_judge else "Inter-annotator agreement was {v:kappa|Cohen's κ per language}. ")
           + f"Against the {who} labels, the automatic "
           "checkers detected hallucinated sentences with {v:detect|precision, recall and F1 of each checker} "
           "(Table IX); the "
           f"correlation of automatic scores with the {who} support rate was " + "{v:hcorr|Spearman ρ per metric}. Table "
           f"VIII gives the {who} hallucination rates. " + "{todo:Say whether the " + who + " labels agree with the automatic "
           "findings for H1 and H2, and which automatic checker is most trustworthy.}", indent=False)
    D.table(("LLM-judge" if R.llm_judge else "Human") + " evaluation: hallucinated summaries (%) and fluency/coverage (1–5)",
            ["Lang", "Track", "Mono", "CM", "Roman", "Flu./Cov."], human_rows(R), [14, 14, 12, 12, 12, 22], size=8)
    D.table(f"Checker vs. {who} labels (hallucinated sentences)", ["Checker", "P", "R", "F1"], detection_rows(R),
            [44, 14, 14, 14], size=8)

    # ---------------- IX. Discussion ----------------
    D.heading("IX", "Discussion")
    D.para("The paired design controls everything except the form of the input: the facts, the reference and the "
           "article are the same across conditions, so a difference in factual consistency between Mono, CM and "
           "Roman can be attributed to how the input is written, up to the small differences in content that the "
           "validation checks cannot catch. Measuring factuality against the original article "
           "keeps the verifiers' own difficulty with code-mixed text out of the comparison; the secondary scores "
           "against the actual input show how much an evaluation that ignores this would mislead. "
           "{todo:Interpret the main findings: is the code-mixing cost larger for Hindi or Bengali, for fine-tuned "
           "models or the LLM, for mixing or for script? Which mechanism of Track B helps most, judging by the error "
           "types? Does LoRA exposure close the gap for mT5?}", indent=False)

    # ---------------- X. Limitations and ethics ----------------
    D.heading("X", "Limitations and Ethics")
    D.para("The code-mixed test inputs are written by an LLM rather than people; the validation checks, the " + ("LLM-judge" if R.llm_judge else "human") + " "
           "check and the small natural set limit but do not remove this concern.{v:rewriter_limit} The automatic checkers make "
           "errors, especially for Bengali, which has no XNLI test set; this is why each checker's precision and "
           "recall against " + ("LLM-judge" if R.llm_judge else "human") + " labels is reported. " + ("The sentence-level evaluation "
           "was done by an LLM judge and not by human annotators, because no Bengali-speaking annotator was "
           "available; an LLM judge can share blind spots with the systems it evaluates, and human annotation with "
           "inter-annotator agreement remains to be done. " if R.llm_judge else "") + "Code-mixing written as English words in the native script "
           "(पुलिस, কোর্ট) is invisible to the script-based CMI, and numbers written as words are not checked. XL-Sum "
           "reference summaries are BBC introductory sentences that sometimes contain facts absent from the article "
           "body, which lowers reference-based scores equally across conditions. The study covers two languages, "
           "one domain (news) and one LLM family. XL-Sum is released under CC BY-NC-SA 4.0 and is used for "
           "non-commercial research; " + ("" if R.llm_judge else "annotators take part voluntarily and ") + "no personal data are collected.",
           indent=False)

    # ---------------- XI. Conclusion ----------------
    D.heading("XI", "Conclusion")
    D.para("FactIndic evaluates factuality-aware summarization of Hindi, Bengali and code-mixed text with paired "
           "inputs that differ only in how they are written, three summarizer families, three prompting strategies "
           "and a multilingual verification layer " + ("compared with an LLM judge. " if R.llm_judge else "checked against human judgements. ")
           +
           "{todo:Two sentences with the answers to H1 and H2 and the main practical recommendation.} Future work "
           "includes naturally occurring code-mixed summaries with references, more Indian languages, and training "
           "summarizers with factuality-aware objectives.", indent=False)

    D.heading("", "Acknowledgment")
    D.para("We thank Dr. Geetanjali Singh for guidance on this project in the CSET346 Natural Language Processing "
           "course at Bennett University" + ("." if R.llm_judge else ", and the annotators for their time."), indent=False)

    # ---------------- References ----------------
    D.heading("", "References")
    for i, key in enumerate(D.cites, 1):
        p = D.doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Mm(6)
        pf.first_line_indent = -Mm(6)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.space_after = D.Pt(1)
        D.rich(p, f"[{i}]\t" + REFS[key], size=8)
        pf.tab_stops.add_tab_stop(Mm(6))
    unused = [k for k in REFS if k not in D.cites]
    if unused:
        print("  references never cited:", unused)
    D.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None, help="the factindic_out folder (omit for a template)")
    ap.add_argument("--out", default="FactIndic_paper.docx")
    a = ap.parse_args()
    print("wrote", build_paper(a.results, a.out))


if __name__ == "__main__":
    main()
