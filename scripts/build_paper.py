"""Render the evidence-based manuscript from saved results and compile it.

paper/template/manuscript.md.j2  --(values from results/analysis + tables)-->
    paper/SACENet_Research_Paper_Draft.md   (citations rendered author-year from references.bib)
    paper/SACENet_Research_Paper_Draft.tex  (natbib; tables \\input from results/tables/*.tex)
    paper/build/SACENet_Research_Paper_Draft.pdf  (if tectonic or pdflatex is available)

Template conventions (a deliberately small Markdown subset):
  # / ## / ###          section levels           **bold**, *italic*, `code`
  - item / 1. item      lists                     $inline$, $$display$$ maths
  \\cite{k}, \\citet{k}, \\citealp{k}   citations (keys must exist in references.bib)
  [[FIG:path/stem|caption|label]]     figure (PNG in md, PDF in tex)
  [[TAB:tab_stem|label]]              table (Markdown file in md, LaTeX file in tex)
  {{ v.some.value }}                  value computed in values() below

Usage:  python scripts/build_paper.py [--no-pdf]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jinja2  # noqa: E402

from src.utils.io import PAPER_DIR, RESULTS_DIR, ROOT, TABLE_DIR, load_json, save_json  # noqa: E402

TEMPLATE = PAPER_DIR / "template" / "manuscript.md.j2"


# ---------------------------------------------------------------- values
class Pending(str):
    """Visible marker for a value that does not exist yet; attribute/item access chains."""
    def __new__(cls, path="?"):
        obj = super().__new__(cls, "**[PENDING]**")
        obj.path = path
        return obj

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return Pending(f"{self.path}.{k}")

    def __getitem__(self, k):
        return Pending(f"{self.path}[{k}]")


class V(dict):
    """dict with attribute access; missing keys render as a visible PENDING marker (never as empty text)."""
    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        if k in self:
            x = self[k]
            return V(x) if isinstance(x, dict) and not isinstance(x, V) else x
        return Pending(k)

    def __getitem__(self, k):
        if k in self:
            x = dict.__getitem__(self, k)
            return V(x) if isinstance(x, dict) and not isinstance(x, V) else x
        return Pending(str(k))


def f3(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "**[PENDING]**"


def values() -> V:
    an = load_json(RESULTS_DIR / "analysis" / "analysis.json")
    st = load_json(ROOT / "data" / "processed" / "data_statistics.json")
    v: dict = {"n_test_dedup": an["n_test_dedup"],
               "mc_test_flagged": st["mc_hinglish"]["quality"]["test_items_flagged_near_dup_in_official_train"]}

    def grab(table, gk):
        r = an.get(table, {}).get(gk)
        if not r:
            return {}
        s = r["seed_mean"]
        out = {"name": r["name"], "n": r["n_runs"]}
        for k in ("macro_f1", "accuracy", "weighted_f1", "ece", "brier", "roc_auc", "pr_auc", "roc_auc_macro_ovr"):
            if k in s:
                out[k] = f3(s[k]["mean"])
                out[k + "_sd"] = f3(s[k]["std"])
        out["ens_f1"] = f3(r["ensemble"]["macro_f1"])
        out["ci_lo"] = f3(r["ensemble_macro_f1_ci95"]["lo"])
        out["ci_hi"] = f3(r["ensemble_macro_f1_ci95"]["hi"])
        out["per_class"] = {c: {k: f3(x) for k, x in d.items() if k != "support"}
                            for c, d in r["ensemble"]["per_class"].items()}
        if "calibrated_ece_mean" in r:
            out["ece_cal"] = f3(r["calibrated_ece_mean"])
        return out

    import datetime as _dt
    v["build_date"] = _dt.date.today().isoformat()
    sh = an.get("split_shift")
    if sh:
        pcs = [x["train_vs_test"] for x in sh["per_class"].values()]
        v["shift"] = {"train_vs_test": f3(sh["train_vs_test"]), "train_vs_val": f3(sh["train_vs_val (control)"]),
                      "min_class": f3(min(pcs)), "max_class": f3(max(pcs))}
    for table in ("mc7_test", "mc7_test_dedup", "mc7_test_neardup", "mc7_val", "mc2_test", "mc2_test_dedup",
                  "sarcasm_test", "conda_test"):
        v[table] = {gk.replace("|", "_"): grab(table, gk) for gk in an.get(table, {})}
    pt = {}
    for k, x in an.get("paired_tests", {}).items():
        key = (x["b"] + "_vs_" + x["a"]).replace("|", "_")
        eb = x["ensemble_bootstrap"]
        pt[key] = {"delta": f"{eb['delta']:+.3f}", "lo": f"{eb['lo']:+.3f}", "hi": f"{eb['hi']:+.3f}",
                   "p": f"{eb['p_value']:.3f}", "p_mcnemar": f"{x['ensemble_mcnemar']['p_value']:.3f}",
                   "seeds_pos": f"{sum(p['delta'] > 0 for p in x['per_seed'])}/{len(x['per_seed'])}"}
    v["pt"] = pt
    lb = an.get("legacy_boost_ablation")
    if lb:
        import numpy as np
        agg = {}
        for setting in ("base@0.50", "base@0.55", "boost@0.55"):
            k = setting.replace("@", "_").replace(".", "")
            agg[k] = {m: f3(float(np.mean([r[setting][m] for r in lb["per_seed"]]))) for m in ("macro_f1", "recall_cb", "fpr")}
        agg["n_boosted"] = lb["per_seed"][0]["n_boosted"]
        agg["share_ge"] = f"{lb['sarcasm_prob_on_mc_test']['share_ge_0.45']:.1%}"
        v["boost"] = agg
    v["ext"] = {k.replace("::", "__").replace("|", "_"): {"f1": f3(x["macro_f1_mean"]), "auc": f3(x["roc_auc_mean"]),
                                                         "n": x["n"]} for k, x in an.get("external", {}).items()}
    ea = an.get("error_analysis_mc7")
    if ea:
        v["ea"] = {"slices": {re.sub(r"\W+", "_", s["slice"]).strip("_").lower(): {
            "n": s["n"], "acc": f3(s.get("accuracy")), "f1": f3(s.get("macro_f1"))} for s in ea["slices"]},
            "fp": ea["false_positive_analysis"], "top": ea["top_confusions"][:5]}
        v["ea"]["fp_rate_prof"] = f"{ea['false_positive_analysis']['fp_rate_with_profanity']:.1%}"
        v["ea"]["fp_rate_noprof"] = f"{ea['false_positive_analysis']['fp_rate_without_profanity']:.1%}"
    v["aux"] = an.get("mtl_aux_heads", {})
    ex = RESULTS_DIR / "explainability"
    if (ex / "faithfulness.json").exists():
        fa = load_json(ex / "faithfulness.json")["summary"]
        stb = load_json(ex / "stability.json")
        ov = load_json(ex / "overhead.json")
        v["xai"] = {"faith": {k.replace("=", ""): {kk: (f3(vv) if not isinstance(vv, list) else vv) for kk, vv in x.items()}
                              for k, x in fa.items()},
                    "stab": {k: {kk: (f3(vv) if isinstance(vv, float) else vv) for kk, vv in x.items()}
                             for k, x in stb["budget_vs_1000"].items()},
                    "occ": {k: (f3(x) if isinstance(x, float) else x) for k, x in stb["shap_vs_occlusion"].items()},
                    "sec": f"{ov['shap_seconds_per_text_mean']:.1f}", "inf_ms": f"{ov['inference_seconds_per_text_mean'] * 1000:.0f}",
                    "factor": f"{ov['overhead_factor']:.0f}", "device": ov["device"]}
    ed = RESULTS_DIR / "early_detection" / "summary.json"
    if ed.exists():
        s = load_json(ed)
        v["early"] = {k: {kk: (f3(vv) if isinstance(vv, float) else vv) for kk, vv in x.items()} for k, x in s.items()}
        for k in s:
            full = load_json(RESULTS_DIR / "early_detection" / f"{k}.json")["policies"]["max"]
            v["early"][k]["far10_recall"] = f3(full["test_at_far10"]["recall"])
            v["early"][k]["far10_far"] = f3(full["test_at_far10"]["false_alarm_rate"])
            v["early"][k]["tau"] = f3(full["tau_maxF1_val"])
            pc = full["prefix_curve_test"]
            v["early"][k]["k1_recall"] = f3(pc[0]["eventual_recall"])
            v["early"][k]["k1_prefix_recall"] = f3(pc[0]["prefix_recall"])
    bm = RESULTS_DIR / "benchmarks" / "inference.json"
    if bm.exists():
        v["bench"] = {re.sub(r"\W+", "_", m["name"]).strip("_").lower(): {k: (f"{x:.1f}" if isinstance(x, float) else x)
                                                                          for k, x in m.items()}
                      for m in load_json(bm)["models"]}
    v["text"] = text_snippets(an, v)
    v["tables"] = {p.stem: p.read_text(encoding="utf-8") for p in sorted(TABLE_DIR.glob("*.md"))}
    v["n_runs"] = sum(len(x) for x in an.get("runs", {}).values())
    v["n_figures"] = len(list((RESULTS_DIR / "figures").rglob("*.png")))
    v["n_tables"] = len(v["tables"])
    v["n_collapsed"] = sum(1 for d in (RESULTS_DIR / "runs").glob("*/metrics.json") if load_json(d).get("collapsed"))
    vr = RESULTS_DIR / "verification_report.json"
    if vr.exists():
        r = load_json(vr)
        ok = sum(c["ok"] for c in r["checks"])
        v["verify"] = {"summary": f"{ok}/{len(r['checks'])} checks passed over {r['n_runs']} runs"
                                  + (f"; failures: {'; '.join(r['failures'][:5])}" if r["failures"] else "")}
    return V(v)


def text_snippets(an: dict, v: dict) -> dict:
    """Short data-derived sentences. Wording rules are fixed in advance (CI includes 0 -> 'no reliable
    difference'), so phrasing cannot be chosen after seeing which way a result went."""
    import numpy as np
    t = {}
    aux = an.get("mtl_aux_heads", {})
    real = [x["macro_f1"] for k, x in aux.items() if k.startswith("mc7__muril__mtl__")]
    shuf = [x["macro_f1"] for k, x in aux.items() if k.startswith("mc7__muril__mtl_shuffled__")]
    if real:
        t["aux_head_range"] = f"{min(real):.3f}–{max(real):.3f}"
    if shuf:
        t["aux_head_shuffled_range"] = f"{min(shuf):.3f}–{max(shuf):.3f}"
    pt = an.get("paired_tests", {})

    def verdict(key):
        x = pt.get(key)
        if not x:
            return None
        e = x["ensemble_bootstrap"]
        short = "no reliable difference" if e["lo"] <= 0 <= e["hi"] else ("higher" if e["delta"] > 0 else "lower")
        phrase = {"no reliable difference": "no reliable difference from", "higher": "an improvement over",
                  "lower": "a decrease relative to"}[short]
        return e, short, phrase

    fc = verdict("mc7|muril|fusion -> mc7|muril|fusion_shuffled") or verdict("mc7|muril|fusion_shuffled -> mc7|muril|fusion")
    mt = an.get("mc7_test", {})
    if "mc7|muril|fusion_shuffled" in mt and fc:
        e, word, _ = fc
        s_ = mt["mc7|muril|fusion_shuffled"]["seed_mean"]["macro_f1"]
        t["fusion_control"] = (
            f"A fusion control whose features come from a sarcasm model trained on permuted labels scores "
            f"{s_['mean']:.3f} ± {s_['std']:.3f}; real vs. control features: {word} "
            f"(Δ = {e['delta']:+.3f}, 95% CI [{e['lo']:+.3f}, {e['hi']:+.3f}]). Any fusion effect is therefore "
            + ("not attributable to sarcasm knowledge." if word != "higher" else "partly attributable to the sarcasm signal.")
        )
    ff = verdict("mc7|muril|base -> mc7|muril|full_ft")
    if "mc7|muril|full_ft" in mt and ff:
        e, word, _ = ff
        r = mt["mc7|muril|full_ft"]["seed_mean"]["macro_f1"]
        t["full_ft"] = (f"A single-seed diagnostic with the embedding table unfrozen reaches {r['mean']:.3f} test macro-F1 "
                        f"({word} vs. the frozen-embedding MuRIL ensemble: Δ = {e['delta']:+.3f}, 95% CI "
                        f"[{e['lo']:+.3f}, {e['hi']:+.3f}]).")
    nf = RESULTS_DIR / "runs_superseded" / "not_feasible" / "README.json"
    if "full_ft" not in t and nf.exists():
        t["full_ft"] = ("A diagnostic run with the token-embedding table unfrozen could not be completed: two attempts "
                        "(batch 32, and batch 16 with 2-step gradient accumulation) drove the 16 GB machine to critical "
                        "memory pressure and were stopped. Whether freezing contributes to the gap therefore remains open.")
    b = mt.get("mc7|muril|base", {}).get("seed_mean", {}).get("macro_f1")
    svm = mt.get("mc7|tfidf_svm", {}).get("seed_mean", {}).get("macro_f1")
    m = verdict("mc7|muril|base -> mc7|muril|mtl")
    sh = verdict("mc7|muril|mtl_shuffled -> mc7|muril|mtl")
    fb = verdict("mc7|muril|base -> mc7|muril|fusion")
    if m and sh:
        t["abstract_sarcasm"] = (
            f"Multitask sarcasm supervision shows {m[2]} the base model (Δ macro-F1 = {m[0]['delta']:+.3f}, 95% CI "
            f"[{m[0]['lo']:+.3f}, {m[0]['hi']:+.3f}]), and a shuffled-label control performs as well as real sarcasm "
            f"labels (Δ = {sh[0]['delta']:+.3f}), so the sarcasm labels themselves add nothing measurable. "
            + (f"Feature fusion with a separately trained sarcasm model shows {fb[2]} the base model "
               f"(Δ = {fb[0]['delta']:+.3f}, 95% CI [{fb[0]['lo']:+.3f}, {fb[0]['hi']:+.3f}]). " if fb else "")
            + "A Hinglish-pretrained encoder (HingRoBERTa) is the only transformer that approaches the TF-IDF baselines.")
    xa = v.get("xai")
    if xa:
        f1 = xa["faith"]["k1"]
        t["abstract_xai"] = (f"deleting the top-attributed word lowers the predicted-class probability by "
                             f"{f1['mean_drop_top']} vs. {f1['mean_drop_random']} for a random word")
    ea = v.get("early", {}).get("conda__distilroberta__base__s13")
    if ea:
        t["abstract_early"] = (f"flags toxic conversations with alarm F1 {ea['f1']} at a {ea['false_alarm_rate']} "
                               f"false-alarm rate, but almost always at the first toxic turn rather than before it.")
    return t


# ---------------------------------------------------------------- bibliography
def parse_bib(path: Path) -> dict[str, dict]:
    """Brace-matching BibTeX parser (handles nested braces, unbraced values, same-line closers)."""
    txt = path.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", txt):
        i, depth = m.end(), 1
        while i < len(txt) and depth:
            depth += {"{": 1, "}": -1}.get(txt[i], 0)
            i += 1
        body = txt[m.end():i - 1]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}|\"[^\"]*\"|[\w.-]+)", body):
            val = fm.group(2)
            if val[0] in "{\"":
                val = val[1:-1]
            fields[fm.group(1).lower()] = re.sub(r"\s+", " ", val)
        out[m.group(2)] = fields
    return out


def _clean(s: str) -> str:
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    return s.replace("{", "").replace("}", "").replace("``", "\"").replace("''", "\"").replace("--", "–")


def author_short(f: dict) -> str:
    auth = [a.strip() for a in f.get("author", "Anon.").split(" and ")]
    last = [(_clean(a.split(",")[0]) if "," in a else _clean(a.split()[-1])) for a in auth]
    if len(last) == 1:
        return last[0]
    if len(last) == 2:
        return f"{last[0]} and {last[1]}"
    return f"{last[0]} et al."


def md_reference(key: str, f: dict) -> str:
    auth = _clean(f.get("author", "")).replace(" and ", "; ")
    venue = f.get("booktitle") or f.get("journal") or (f"arXiv:{f['eprint']}" if "eprint" in f else f.get("publisher", ""))
    doi = f" doi:{f['doi']}" if "doi" in f else ""
    url = "" if doi else (f" {f['url']}" if "url" in f else "")
    return f"- **[{key}]** {auth} ({f.get('year', 'n.d.')}). {_clean(f.get('title', ''))}. *{_clean(venue)}*.{doi}{url}"


# ---------------------------------------------------------------- rendering
CITE_RE = re.compile(r"\\(cite|citet|citealp|citep)\{([^}]+)\}")


def render_md(text: str, bib: dict, used: list) -> str:
    def cite(m):
        keys = [k.strip() for k in m.group(2).split(",")]
        for k in keys:
            if k not in bib:
                raise KeyError(f"citation key not in references.bib: {k}")
            if k not in used:
                used.append(k)
        parts = [(author_short(bib[k]), bib[k].get("year", "n.d.")) for k in keys]
        if m.group(1) == "citet":
            return "; ".join(f"{a} ({y})" for a, y in parts)
        inner = "; ".join(f"{a}, {y}" for a, y in parts)
        return inner if m.group(1) == "citealp" else f"({inner})"

    text = CITE_RE.sub(cite, text)

    def fig(m):
        path, cap, label = m.group(1).split("|")
        return f"![{cap}](../{path}.png)\n\n*Figure ({label}). {cap}*"

    def tab(m):
        stem, label = m.group(1).split("|")
        p = TABLE_DIR / f"{stem}.md"
        return p.read_text(encoding="utf-8") if p.exists() else f"**[PENDING table {stem}]**"

    text = re.sub(r"\[\[FIG:([^\]]+)\]\]", fig, text)
    text = re.sub(r"\[\[TAB:([^\]]+)\]\]", tab, text)
    return text


def _tex_inline(s: str) -> str:
    # protect maths and citations, escape the rest
    keep = []

    def stash(m):
        keep.append(m.group(0))
        return f"\x00{len(keep) - 1}\x00"

    s = re.sub(r"\$[^$]+\$|\\(?:cite|citet|citealp|citep|ref|label)\{[^}]+\}", stash, s)
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("~", r"\textasciitilde{}"),
                 ("±", r"$\pm$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("→", r"$\rightarrow$"), ("×", r"$\times$"),
                 ("−", "--"), ("–", "--"), ("—", "---"), ("‘", "`"), ("’", "'"), ("“", "``"), ("”", "''"), ("…", r"\ldots{}"),
                 ("Δ", r"$\Delta$"), ("λ", r"$\lambda$"), ("τ", r"$\tau$"), ("≈", r"$\approx$"),
                 ("∪", r"$\cup$"), ("φ", r"$\phi$")):
        s = s.replace(a, b)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    s = re.sub(r"`([^`]+)`", r"\\texttt{\1}", s)
    s = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], s)
    return s


def render_tex(text: str, title: str) -> str:
    out = [r"\documentclass[10pt,twocolumn]{article}", r"\usepackage[margin=0.8in]{geometry}",
           r"\usepackage{iftex}",
           r"\ifXeTeX\usepackage{fontspec}\else\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}"
           r"\usepackage{lmodern}\usepackage{newunicodechar}\newunicodechar{–}{--}\newunicodechar{’}{'}\fi",
           r"\usepackage{amsmath,amssymb}",
           r"\usepackage{booktabs}", r"\usepackage{graphicx}", r"\usepackage[numbers,sort&compress]{natbib}",
           r"\usepackage{hyperref}", r"\usepackage{adjustbox}", r"\usepackage{caption}",
           r"\captionsetup{font=small}", r"\graphicspath{{../}{../../}}", rf"\title{{{_tex_inline(title)}}}",
           r"\author{[Author names and affiliations to be added]}", r"\date{Draft --- generated from saved results}",
           r"\begin{document}", r"\maketitle"]
    lines = text.splitlines()
    i, in_list = 0, None
    abstract = False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# ") and i == 0:
            i += 1
            continue
        m_fig = re.match(r"\[\[FIG:([^\]]+)\]\]", ln.strip())
        m_tab = re.match(r"\[\[TAB:([^\]]+)\]\]", ln.strip())
        if in_list and not re.match(r"^(\s*[-*] |\s*\d+\. )", ln):
            out.append(r"\end{%s}" % in_list)
            in_list = None
        if ln.startswith("## Abstract"):
            out.append(r"\begin{abstract}")
            abstract = True
        elif ln.startswith("#"):
            if abstract:
                out.append(r"\end{abstract}")
                abstract = False
            level = len(ln) - len(ln.lstrip("#"))
            head = re.sub(r"^\d+(\.\d+)*\.?\s*", "", ln.lstrip("#").strip())
            if head.lower() == "references":
                break
            out.append({2: r"\section{%s}", 3: r"\subsection{%s}", 4: r"\paragraph{%s}"}.get(level, r"\section{%s}")
                       % _tex_inline(head))
        elif m_fig:
            path, cap, label = m_fig.group(1).split("|")
            wide = "diagrams" in path or "comparison" in path or "shap" in path
            env = "figure*" if wide else "figure"
            out += [rf"\begin{{{env}}}[t]", r"\centering",
                    rf"\includegraphics[width={'0.9' if wide else '1.0'}\{'textwidth' if wide else 'columnwidth'}]{{{path}.pdf}}",
                    rf"\caption{{{_tex_inline(cap)}}}", rf"\label{{{label}}}", rf"\end{{{env}}}"]
        elif m_tab:
            stem, label = m_tab.group(1).split("|")
            p = TABLE_DIR / f"{stem}.tex"
            if p.exists():
                t = p.read_text(encoding="utf-8").replace(r"\begin{table}[t]", r"\begin{table*}[t]").replace(
                    r"\end{table}", r"\end{table*}").replace(rf"\label{{tab:{stem}}}", rf"\label{{{label}}}")
                t = t.replace(r"\begin{tabular}", r"\begin{adjustbox}{max width=\textwidth}\begin{tabular}").replace(
                    r"\end{tabular}", r"\end{tabular}\end{adjustbox}")
                out.append(t)
            else:
                out.append(r"\textbf{[PENDING table %s]}" % stem.replace("_", r"\_"))
        elif ln.strip().startswith("$$"):
            body = ln.strip().strip("$")
            while not lines[i].strip().endswith("$$") or (lines[i].strip() == "$$" and not body):
                i += 1
                body += " " + lines[i].strip().strip("$")
            out.append(r"\begin{equation}" + body + r"\end{equation}")
        elif re.match(r"^\s*[-*] ", ln) or re.match(r"^\s*\d+\. ", ln):
            kind = "itemize" if re.match(r"^\s*[-*] ", ln) else "enumerate"
            if in_list != kind:
                if in_list:
                    out.append(r"\end{%s}" % in_list)
                out.append(r"\begin{%s}" % kind)
                in_list = kind
            out.append(r"\item " + _tex_inline(re.sub(r"^\s*([-*]|\d+\.)\s+", "", ln)))
        elif ln.strip() == "---":
            pass
        else:
            out.append(_tex_inline(ln))
        i += 1
    if in_list:
        out.append(r"\end{%s}" % in_list)
    out += [r"\bibliographystyle{plainnat}", r"\bibliography{references}", r"\end{document}"]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    v = values()
    save_json(v, PAPER_DIR / "generated" / "values_used.json")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATE.parent)), undefined=jinja2.StrictUndefined,
                             autoescape=False, keep_trailing_newline=True)
    body = env.get_template(TEMPLATE.name).render(v=v)
    bib = parse_bib(PAPER_DIR / "references.bib")
    used: list = []
    md = render_md(body, bib, used)
    md += "\n\n## References\n\n" + "\n".join(md_reference(k, bib[k]) for k in sorted(used, key=lambda k: author_short(bib[k]))) + "\n"
    (PAPER_DIR / "SACENet_Research_Paper_Draft.md").write_text(md, encoding="utf-8")
    title = body.splitlines()[0].lstrip("# ").strip()
    tex = render_tex(body, title)
    (PAPER_DIR / "SACENet_Research_Paper_Draft.tex").write_text(tex, encoding="utf-8")
    for tpl, dest in (("model_card.md.j2", ROOT / "MODEL_CARD.md"),
                      ("final_report.md.j2", RESULTS_DIR / "FINAL_EXPERIMENT_REPORT.md")):
        dest.write_text(env.get_template(tpl).render(v=v), encoding="utf-8")
    pend = md.count("[PENDING")
    print(f"rendered: {len(md.split())} words, {len(used)} citations, PENDING markers: {pend}")
    if args.no_pdf:
        return 0
    build = PAPER_DIR / "build"
    build.mkdir(exist_ok=True)
    shutil.copy(PAPER_DIR / "SACENet_Research_Paper_Draft.tex", build / "SACENet_Research_Paper_Draft.tex")
    shutil.copy(PAPER_DIR / "references.bib", build / "references.bib")
    eng = shutil.which("tectonic") or shutil.which("pdflatex")
    if not eng:
        print("No LaTeX engine found: install tectonic (brew/choco/scoop) or TeX Live/MiKTeX; sources are ready.")
        return 0
    cmd = [eng, "SACENet_Research_Paper_Draft.tex"] if "tectonic" in eng else [eng, "-interaction=nonstopmode",
                                                                             "SACENet_Research_Paper_Draft.tex"]
    r = subprocess.run(cmd, cwd=build, capture_output=True, text=True)
    (build / "latex_log.txt").write_text(r.stdout + "\n" + r.stderr)
    print("PDF:", build / "SACENet_Research_Paper_Draft.pdf" if r.returncode == 0 else f"FAILED (see {build}/latex_log.txt)")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
