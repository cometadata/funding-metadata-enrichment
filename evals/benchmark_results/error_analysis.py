"""Compare new config predictions against gold test.jsonl and output per-document error analysis."""

import json
import sys
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz

THRESHOLD = 0.8
FUNDER_THRESHOLD = 0.8

GOLD_PATH = Path("/Users/adambuttrick/Downloads/test.jsonl")
PRED_BASE = Path("/Users/adambuttrick/Downloads/funding-extraction-harness-benchmark")
OUT_DIR = Path("/Users/adambuttrick/Documents/GitHub/cometadata/funding-metadata-enrichment/evals/benchmark_results")

NEW_CONFIGS = [
    "funding-parsing-lora-Qwen3_8B-ep2-r64-a128-synthetic",
    "funding-parsing-lora-Qwen3_8B-ep2-r64-a128-synthetic-thinking",
    "funding-parsing-lora-Qwen3_8B-ep2p1-r64-a128-synthetic-twostage",
    "funding-parsing-lora-Qwen3_8B-ep2p1-r64-a128-synthetic-twostage-thinking",
]


def normalize_text(s):
    if not s:
        return ""
    s = s.strip().lower()
    s = " ".join(s.split())
    return s


def sim(a, b):
    na, nb = normalize_text(a), normalize_text(b)
    if not na or not nb:
        return 0.0
    scores = [
        fuzz.partial_ratio(na, nb),
        fuzz.partial_ratio(nb, na),
        fuzz.token_sort_ratio(na, nb),
        fuzz.token_set_ratio(na, nb),
    ]
    return max(scores) / 100.0


def normalize_id(s):
    return s.strip().upper().replace("-", "").replace("/", "").replace(" ", "")


def load_gold():
    docs = {}
    with open(GOLD_PATH) as f:
        for line in f:
            row = json.loads(line)
            docs[row["doi"]] = row
    return docs


def load_preds(config_name):
    path = PRED_BASE / config_name / "test-00000-of-00001.jsonl"
    docs = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            docs[row["doi"]] = row
    return docs


def flatten_awards(funders, scheme_key="funding_scheme", title_key="award_title"):
    """Return flat lists: funder_names, award_ids, schemes, titles."""
    names, ids, schemes, titles = [], [], [], []
    for f in funders:
        name = f.get("funder_name", "") or ""
        if name:
            names.append(name)
        for a in f.get("awards", []):
            ids.extend(a.get("award_ids", []))
            schemes.extend(a.get(scheme_key, []))
            titles.extend(a.get(title_key, []))
    return names, ids, schemes, titles


def greedy_match_strings(gold_items, pred_items, threshold):
    """Return (matched_pairs, unmatched_gold, unmatched_pred)."""
    if not gold_items or not pred_items:
        return [], list(range(len(gold_items))), list(range(len(pred_items)))

    scored = []
    for gi, g in enumerate(gold_items):
        for pi, p in enumerate(pred_items):
            s = sim(g, p)
            if s >= threshold:
                scored.append((s, gi, pi))
    scored.sort(key=lambda x: -x[0])

    used_g, used_p = set(), set()
    matched = []
    for s, gi, pi in scored:
        if gi not in used_g and pi not in used_p:
            matched.append((gi, pi, s))
            used_g.add(gi)
            used_p.add(pi)

    unmatched_g = [i for i in range(len(gold_items)) if i not in used_g]
    unmatched_p = [i for i in range(len(pred_items)) if i not in used_p]
    return matched, unmatched_g, unmatched_p


def greedy_match_ids(gold_ids, pred_ids):
    g_norm = [normalize_id(x) for x in gold_ids]
    p_norm = [normalize_id(x) for x in pred_ids]

    used_g, used_p = set(), set()
    matched = []
    for gi, gn in enumerate(g_norm):
        for pi, pn in enumerate(p_norm):
            if gi not in used_g and pi not in used_p and gn == pn:
                matched.append((gi, pi))
                used_g.add(gi)
                used_p.add(pi)

    unmatched_g = [i for i in range(len(gold_ids)) if i not in used_g]
    unmatched_p = [i for i in range(len(pred_ids)) if i not in used_p]
    return matched, unmatched_g, unmatched_p


def analyze_document(gold_doc, pred_doc):
    """Return a dict describing all errors for this document."""
    errors = []

    g_names, g_ids, g_schemes, g_titles = flatten_awards(gold_doc.get("funders", []))
    p_names, p_ids, p_schemes, p_titles = flatten_awards(pred_doc.get("funders", []))

    # --- Funder matching ---
    f_matched, f_miss_g, f_miss_p = greedy_match_strings(g_names, p_names, FUNDER_THRESHOLD)

    for gi in f_miss_g:
        errors.append({
            "level": "funder",
            "type": "missing",
            "gold": g_names[gi],
            "predicted": None,
            "detail": f"Gold funder not found in predictions",
        })
    for pi in f_miss_p:
        errors.append({
            "level": "funder",
            "type": "spurious",
            "gold": None,
            "predicted": p_names[pi],
            "detail": f"Predicted funder not in gold",
        })
    for gi, pi, score in f_matched:
        if score < 1.0:
            errors.append({
                "level": "funder",
                "type": "name_mismatch",
                "gold": g_names[gi],
                "predicted": p_names[pi],
                "detail": f"Matched but names differ (sim={score:.3f})",
            })

    # --- Award ID matching ---
    id_matched, id_miss_g, id_miss_p = greedy_match_ids(g_ids, p_ids)

    for gi in id_miss_g:
        errors.append({
            "level": "award_id",
            "type": "missing",
            "gold": g_ids[gi],
            "predicted": None,
            "detail": "Gold award ID not found in predictions",
        })
    for pi in id_miss_p:
        errors.append({
            "level": "award_id",
            "type": "spurious",
            "gold": None,
            "predicted": p_ids[pi],
            "detail": "Predicted award ID not in gold",
        })

    # --- Funding scheme matching ---
    s_matched, s_miss_g, s_miss_p = greedy_match_strings(g_schemes, p_schemes, THRESHOLD)

    for gi in s_miss_g:
        errors.append({
            "level": "funding_scheme",
            "type": "missing",
            "gold": g_schemes[gi],
            "predicted": None,
            "detail": "Gold funding scheme not found in predictions",
        })
    for pi in s_miss_p:
        errors.append({
            "level": "funding_scheme",
            "type": "spurious",
            "gold": None,
            "predicted": p_schemes[pi],
            "detail": "Predicted funding scheme not in gold",
        })

    # --- Award title matching ---
    t_matched, t_miss_g, t_miss_p = greedy_match_strings(g_titles, p_titles, THRESHOLD)

    for gi in t_miss_g:
        errors.append({
            "level": "award_title",
            "type": "missing",
            "gold": g_titles[gi],
            "predicted": None,
            "detail": "Gold award title not found in predictions",
        })
    for pi in t_miss_p:
        errors.append({
            "level": "award_title",
            "type": "spurious",
            "gold": None,
            "predicted": p_titles[pi],
            "detail": "Predicted award title not in gold",
        })

    return errors


def main():
    gold_docs = load_gold()
    print(f"Loaded {len(gold_docs)} gold documents")

    for config_name in NEW_CONFIGS:
        print(f"\nProcessing {config_name}...")
        pred_docs = load_preds(config_name)

        output = []
        error_summary = defaultdict(lambda: defaultdict(int))

        for doi, gold in gold_docs.items():
            pred = pred_docs.get(doi, {"funders": []})
            errors = analyze_document(gold, pred)

            record = {
                "doi": doi,
                "funding_statement": gold.get("funding_statement", ""),
                "gold_funders": gold.get("funders", []),
                "predicted_funders": pred.get("funders", []),
                "error_count": len(errors),
                "errors": errors,
            }
            output.append(record)

            for e in errors:
                error_summary[e["level"]][e["type"]] += 1

        # Sort: most errors first
        output.sort(key=lambda x: -x["error_count"])

        out_path = OUT_DIR / f"{config_name}_error_analysis.jsonl"
        with open(out_path, "w") as f:
            for record in output:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        total_errors = sum(r["error_count"] for r in output)
        docs_with_errors = sum(1 for r in output if r["error_count"] > 0)
        print(f"  Output: {out_path.name}")
        print(f"  {docs_with_errors}/{len(output)} docs with errors, {total_errors} total errors")
        print(f"  Error breakdown:")
        for level in ["funder", "award_id", "funding_scheme", "award_title"]:
            counts = error_summary[level]
            if counts:
                parts = ", ".join(f"{t}: {c}" for t, c in sorted(counts.items()))
                print(f"    {level}: {parts}")


if __name__ == "__main__":
    main()
