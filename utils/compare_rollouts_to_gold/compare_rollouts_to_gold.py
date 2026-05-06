import os
import re
import csv
import json
import argparse
from collections import Counter, defaultdict

import ftfy
from rapidfuzz import fuzz

DEFAULT_FUZZY_THRESHOLD = 90
DEFAULT_OUTPUT_DIR = "error_analysis"

ERROR_VALUE_KEYS = (
    "missing_funders",
    "extra_funders",
    "missing_award_ids",
    "extra_award_ids",
    "missing_schemes",
    "extra_schemes",
    "missing_award_titles",
    "extra_award_titles",
)

ERROR_TYPE_TO_VALUE_KEY = {
    "missing_funder": ("missing_funders", "gold_value"),
    "extra_funder": ("extra_funders", "pred_value"),
    "missing_award_id": ("missing_award_ids", "gold_value"),
    "extra_award_id": ("extra_award_ids", "pred_value"),
    "missing_scheme": ("missing_schemes", "gold_value"),
    "extra_scheme": ("extra_schemes", "pred_value"),
    "missing_award_title": ("missing_award_titles", "gold_value"),
    "extra_award_title": ("extra_award_titles", "pred_value"),
}

SIDE_BY_SIDE_FIELDS = [
    "doi", "funding_statement", "gold_funders", "pred_funders",
    "error_count", "error_types",
    *ERROR_VALUE_KEYS,
    "error_details",
]


def normalize_text(text):
    if text is None:
        return ""
    text = ftfy.fix_text(text)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("``", '"').replace("''", '"')
    text = text.replace("\u22c6", "").replace("\u2022", "").replace("*", "")
    text = " ".join(text.split())
    text = re.sub(r"\\([a-z])", r"\1", text)
    return text.strip().lower()


def normalize_set(values):
    if not values:
        return set()
    return {normalize_text(v) for v in values if v}


def fuzzy_similarity(a, b):
    na, nb = normalize_text(a), normalize_text(b)
    if na == nb:
        return 100.0
    return max(
        fuzz.ratio(na, nb),
        fuzz.partial_ratio(na, nb),
        fuzz.partial_ratio(nb, na),
        fuzz.token_sort_ratio(na, nb),
        fuzz.token_set_ratio(na, nb),
    )


def load_predictions(path):
    preds, parse_errors = {}, []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            record = json.loads(line)
            doi = record.get("doi")
            if not doi:
                continue
            assistant_msg = next(
                (m["content"] for m in record.get("messages", [])
                 if m["role"] == "assistant"),
                None,
            )
            if assistant_msg is None:
                parse_errors.append(
                    {"line": lineno, "doi": doi, "error": "No assistant message"})
                continue
            try:
                preds[doi] = json.loads(assistant_msg)
            except json.JSONDecodeError as exc:
                parse_errors.append(
                    {"line": lineno, "doi": doi, "error": f"JSON parse error: {exc}"})
    return preds, parse_errors


def load_gold(path):
    gold = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            doi = record.get("doi")
            if doi:
                gold[doi] = {
                    "funding_statement": record.get("funding_statement", ""),
                    "funders": record.get("funders", []),
                }
    return gold


def fuzzy_match_sets(gold_set, pred_set, threshold):
    if not gold_set and not pred_set:
        return set(), set(), set(), set()

    scored = sorted(
        ((fuzzy_similarity(g, p), g, p) for g in gold_set for p in pred_set),
        key=lambda x: -x[0],
    )

    matched_gold, matched_pred = set(), set()
    for score, g, p in scored:
        if score < threshold:
            break
        if g not in matched_gold and p not in matched_pred:
            matched_gold.add(g)
            matched_pred.add(p)

    return (matched_gold, matched_pred,
            gold_set - matched_gold, pred_set - matched_pred)


def diff_sets(gold_set, pred_set, use_fuzzy, threshold):
    if use_fuzzy:
        _, _, missing, extra = fuzzy_match_sets(gold_set, pred_set, threshold)
        return missing, extra
    return gold_set - pred_set, pred_set - gold_set


def flatten_awards(funders_list):
    rows = []
    for funder in funders_list:
        fname = funder.get("funder_name") or None
        awards = funder.get("awards") or [
            {"funding_scheme": [], "award_ids": [], "award_title": []}
        ]
        for award in awards:
            rows.append({
                "funder_name": fname,
                "funding_scheme": award.get("funding_scheme") or [],
                "award_ids": award.get("award_ids") or [],
                "award_title": award.get("award_title") or [],
            })
    return rows


def _make_error(doi, stmt, error_type, field, gold_value, pred_value, detail):
    return {
        "doi": doi,
        "funding_statement": stmt,
        "error_type": error_type,
        "field": field,
        "gold_value": gold_value,
        "pred_value": pred_value,
        "detail": detail,
    }


def _diff_field_errors(doi, stmt, gold_vals, pred_vals,
                       field, missing_type, extra_type, funder_display,
                       use_fuzzy, threshold):
    errors = []
    missing, extra = diff_sets(gold_vals, pred_vals, use_fuzzy, threshold)
    for v in missing:
        errors.append(_make_error(
            doi, stmt, missing_type, field, v, "",
            f"{field.replace('_', ' ').title()} '{v}' for funder "
            f"'{funder_display}' in gold but missing from prediction"))
    for v in extra:
        errors.append(_make_error(
            doi, stmt, extra_type, field, "", v,
            f"{field.replace('_', ' ').title()} '{v}' for funder "
            f"'{funder_display}' predicted but not in gold"))
    return errors


def compare_record(doi, pred_funders, gold_funders, funding_statement,
                   use_fuzzy=False, threshold=DEFAULT_FUZZY_THRESHOLD):
    pred_flat = flatten_awards(pred_funders)
    gold_flat = flatten_awards(gold_funders)
    errors = []
    stmt = funding_statement

    pred_names = {normalize_text(r["funder_name"]) for r in pred_flat if r["funder_name"]}
    gold_names = {normalize_text(r["funder_name"]) for r in gold_flat if r["funder_name"]}

    missing_f, extra_f = diff_sets(gold_names, pred_names, use_fuzzy, threshold)
    for f in missing_f:
        errors.append(_make_error(
            doi, stmt, "missing_funder", "funder_name", f, "",
            f"Funder '{f}' present in gold but missing from prediction"))
    for f in extra_f:
        errors.append(_make_error(
            doi, stmt, "extra_funder", "funder_name", "", f,
            f"Funder '{f}' predicted but not in gold"))

    pred_null = sum(1 for r in pred_flat if r["funder_name"] is None)
    gold_null = sum(1 for r in gold_flat if r["funder_name"] is None)
    if pred_null != gold_null:
        errors.append(_make_error(
            doi, stmt, "null_funder_count_mismatch", "funder_name",
            f"{gold_null} null funders", f"{pred_null} null funders",
            f"Gold has {gold_null} null-funder entries, prediction has {pred_null}"))

    def _award_map(flat):
        m = defaultdict(list)
        for r in flat:
            key = normalize_text(r["funder_name"]) if r["funder_name"] else "__NULL__"
            m[key].append(r)
        return m

    pred_map, gold_map = _award_map(pred_flat), _award_map(gold_flat)

    if use_fuzzy:
        matched_g, _, _, _ = fuzzy_match_sets(gold_names, pred_names, threshold)
        funder_pairs = []
        remaining_p = set(pred_names)
        for g in matched_g:
            best_p = max(remaining_p, key=lambda p: fuzzy_similarity(g, p))
            funder_pairs.append((g, best_p))
            remaining_p.discard(best_p)
    else:
        funder_pairs = [(f, f) for f in gold_names & pred_names]

    if pred_null > 0 and gold_null > 0:
        funder_pairs.append(("__NULL__", "__NULL__"))

    field_specs = [
        ("award_ids", "award_ids", "missing_award_id", "extra_award_id"),
        ("funding_scheme", "funding_scheme", "missing_scheme", "extra_scheme"),
        ("award_title", "award_title", "missing_award_title", "extra_award_title"),
    ]

    for gold_key, pred_key in funder_pairs:
        display = gold_key if gold_key != "__NULL__" else "(null funder)"
        g_awards = gold_map.get(gold_key, [])
        p_awards = pred_map.get(pred_key, [])

        for src_field, field_label, miss_type, extra_type in field_specs:
            g_vals = normalize_set([v for a in g_awards for v in a[src_field]])
            p_vals = normalize_set([v for a in p_awards for v in a[src_field]])
            errors.extend(_diff_field_errors(
                doi, stmt, g_vals, p_vals,
                field_label, miss_type, extra_type, display,
                use_fuzzy, threshold))

        if len(p_awards) != len(g_awards):
            errors.append(_make_error(
                doi, stmt, "award_count_mismatch", "awards",
                str(len(g_awards)), str(len(p_awards)),
                f"Funder '{display}': gold has {len(g_awards)} award entries, "
                f"prediction has {len(p_awards)}"))

    if len(pred_flat) != len(gold_flat):
        errors.append(_make_error(
            doi, stmt, "funder_count_mismatch", "funders",
            str(len(gold_flat)), str(len(pred_flat)),
            f"Gold has {len(gold_flat)} funder-award entries total, "
            f"prediction has {len(pred_flat)}"))

    return errors


def run_comparison(common_dois, preds, gold, use_fuzzy, threshold):
    all_errors, per_doi_summary = [], []
    perfect_count = 0
    for doi in sorted(common_dois):
        errs = compare_record(
            doi, preds[doi], gold[doi]["funders"],
            gold[doi]["funding_statement"],
            use_fuzzy=use_fuzzy, threshold=threshold)
        all_errors.extend(errs)
        if not errs:
            perfect_count += 1
        per_doi_summary.append({
            "doi": doi,
            "error_count": len(errs),
            "error_types": sorted(set(e["error_type"] for e in errs)),
            "funding_statement": gold[doi]["funding_statement"],
        })
    return all_errors, per_doi_summary, perfect_count


def _extract_error_values(doi_errors):
    vals = {k: [] for k in ERROR_VALUE_KEYS}
    for e in doi_errors:
        mapping = ERROR_TYPE_TO_VALUE_KEY.get(e["error_type"])
        if mapping:
            bucket, source = mapping
            vals[bucket].append(e[source])
    return vals


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_side_by_side(all_errors, common_dois, gold, preds):
    errors_by_doi = defaultdict(list)
    for e in all_errors:
        errors_by_doi[e["doi"]].append(e)

    rows = []
    for doi in sorted(common_dois):
        doi_errors = errors_by_doi.get(doi, [])
        if not doi_errors:
            continue
        vals = _extract_error_values(doi_errors)
        row = {
            "doi": doi,
            "funding_statement": gold[doi]["funding_statement"],
            "gold_funders": json.dumps(gold[doi]["funders"], ensure_ascii=False),
            "pred_funders": json.dumps(preds[doi], ensure_ascii=False),
            "error_count": len(doi_errors),
            "error_types": "; ".join(sorted(set(e["error_type"] for e in doi_errors))),
            "error_details": "; ".join(e["detail"] for e in doi_errors),
        }
        for k in ERROR_VALUE_KEYS:
            row[k] = "; ".join(vals[k])
        rows.append(row)
    return rows


def write_outputs(output_dir, all_errors, per_doi_summary, common_dois,
                  gold, preds, total_compared, perfect_count, parse_errors,
                  mode_label, pred_file, gold_file, threshold,
                  pred_count, gold_count, pred_only, gold_only):
    os.makedirs(output_dir, exist_ok=True)
    error_dois = total_compared - perfect_count

    error_type_counts = Counter(e["error_type"] for e in all_errors)
    field_counts = Counter(e["field"] for e in all_errors)
    n_missing = sum(1 for e in all_errors if e["error_type"].startswith("missing_"))
    n_extra = sum(1 for e in all_errors if e["error_type"].startswith("extra_"))
    n_mismatch = sum(1 for e in all_errors if e["error_type"].endswith("_mismatch"))

    error_fields = ["doi", "error_type", "field", "gold_value", "pred_value",
                    "detail", "funding_statement"]
    _write_csv(os.path.join(output_dir, "all_errors.csv"), error_fields, all_errors)
    _write_jsonl(os.path.join(output_dir, "all_errors.jsonl"), all_errors)

    sorted_summary = sorted(per_doi_summary, key=lambda x: -x["error_count"])
    csv_summary = [
        {**r, "error_types": "; ".join(r["error_types"])} for r in sorted_summary
    ]
    _write_csv(os.path.join(output_dir, "per_doi_summary.csv"),
               ["doi", "error_count", "error_types", "funding_statement"], csv_summary)
    _write_jsonl(os.path.join(output_dir, "per_doi_summary.jsonl"), sorted_summary)

    type_rows = []
    for etype, count in error_type_counts.most_common():
        affected = len({e["doi"] for e in all_errors if e["error_type"] == etype})
        type_rows.append({
            "error_type": etype,
            "count": count,
            "pct_of_all_errors": f"{100 * count / len(all_errors):.1f}" if all_errors else "0",
            "dois_affected": affected,
            "pct_of_dois": f"{100 * affected / total_compared:.1f}",
        })
    _write_csv(os.path.join(output_dir, "error_type_summary.csv"),
               ["error_type", "count", "pct_of_all_errors", "dois_affected", "pct_of_dois"],
               type_rows)

    sbs_rows = _build_side_by_side(all_errors, common_dois, gold, preds)
    _write_csv(os.path.join(output_dir, "side_by_side_errors.csv"),
               SIDE_BY_SIDE_FIELDS, sbs_rows)

    sbs_jsonl_rows = []
    for row in sbs_rows:
        obj = dict(row)
        obj["gold_funders"] = json.loads(obj["gold_funders"])
        obj["pred_funders"] = json.loads(obj["pred_funders"])
        for k in (*ERROR_VALUE_KEYS, "error_types", "error_details"):
            obj[k] = [s for s in obj[k].split("; ") if s]
        sbs_jsonl_rows.append(obj)
    _write_jsonl(os.path.join(output_dir, "side_by_side_errors.jsonl"), sbs_jsonl_rows)

    if parse_errors:
        _write_jsonl(os.path.join(output_dir, "parse_errors.jsonl"), parse_errors)

    is_fuzzy = "fuzzy" in mode_label.lower()
    report_lines = [
        "=" * 70,
        f"FUNDING ANNOTATION ERROR ANALYSIS REPORT ({mode_label})",
        f"Predictions: {pred_file}",
        f"Gold standard: {gold_file}",
        *([ f"Fuzzy threshold: {threshold}"] if is_fuzzy else []),
        "=" * 70, "",
        "DOI COVERAGE",
        f"  Prediction DOIs: {pred_count}",
        f"  Gold DOIs: {gold_count}",
        f"  Common (compared): {total_compared}",
        f"  Prediction-only: {len(pred_only)}",
        f"  Gold-only: {len(gold_only)}", "",
        "OVERALL ACCURACY",
        f"  Total DOIs compared: {total_compared}",
        f"  Perfect matches: {perfect_count} ({100 * perfect_count / total_compared:.1f}%)",
        f"  DOIs with errors: {error_dois} ({100 * error_dois / total_compared:.1f}%)",
        f"  Total individual errors: {len(all_errors)}", "",
        "ERROR TYPE BREAKDOWN",
    ]
    for etype, count in error_type_counts.most_common():
        pct = 100 * count / len(all_errors) if all_errors else 0
        da = len({e["doi"] for e in all_errors if e["error_type"] == etype})
        report_lines.append(
            f"  {etype}: {count} errors ({pct:.1f}%), "
            f"affects {da} DOIs ({100 * da / total_compared:.1f}%)")
    report_lines += [
        "", "ERRORS BY FIELD",
        *[f"  {fld}: {cnt} ({100 * cnt / len(all_errors):.1f}%)"
          if all_errors else f"  {fld}: 0"
          for fld, cnt in field_counts.most_common()],
        "", "ERROR DIRECTION",
        f"  Missing (gold has, pred doesn't): {n_missing}",
        f"  Extra (pred has, gold doesn't): {n_extra}",
        f"  Count mismatches: {n_mismatch}", "",
        "OUTPUT FILES",
        "  all_errors.csv / .jsonl        - Every individual error",
        "  per_doi_summary.csv / .jsonl    - Error count per DOI",
        "  error_type_summary.csv          - Counts by error type",
        "  side_by_side_errors.csv / .jsonl - Gold vs pred with specific error values",
    ]
    with open(os.path.join(output_dir, "summary_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines) + "\n")

    print(f"\n  [{mode_label}] DOIs compared: {total_compared}")
    print(f"  [{mode_label}] Perfect matches: {perfect_count} "
          f"({100 * perfect_count / total_compared:.1f}%)")
    print(f"  [{mode_label}] DOIs with errors: {error_dois} "
          f"({100 * error_dois / total_compared:.1f}%)")
    print(f"  [{mode_label}] Total individual errors: {len(all_errors)}")
    print(f"\n  [{mode_label}] Error Type Breakdown:")
    for etype, count in error_type_counts.most_common():
        da = len({e["doi"] for e in all_errors if e["error_type"] == etype})
        print(f"    {etype}: {count} errors, {da} DOIs ({100 * da / total_compared:.1f}%)")
    print(f"\n  [{mode_label}] Error Direction:")
    print(f"    Missing: {n_missing}")
    print(f"    Extra:   {n_extra}")
    print(f"    Count mismatches: {n_mismatch}")
    print(f"\n  [{mode_label}] Wrote files to {output_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare SFT rollout predictions against gold funding annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p", "--predictions", required=True,
        help="Path to predictions JSONL (SFT chat format with system/user/assistant).",
    )
    parser.add_argument(
        "-g", "--gold", required=True,
        help="Path to gold annotations JSONL.",
    )
    parser.add_argument(
        "-o", "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Root output directory (strict/ and fuzzy/ subdirs are created inside).",
    )
    parser.add_argument(
        "-t", "--threshold", type=int, default=DEFAULT_FUZZY_THRESHOLD,
        help="Fuzzy similarity threshold (0-100).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading predictions...")
    preds, parse_errors = load_predictions(args.predictions)
    print(f"  Loaded {len(preds)} predictions, {len(parse_errors)} parse errors")

    print("Loading gold annotations...")
    gold = load_gold(args.gold)
    print(f"  Loaded {len(gold)} gold records")

    pred_dois = set(preds)
    gold_dois = set(gold)
    common_dois = pred_dois & gold_dois
    pred_only = pred_dois - gold_dois
    gold_only = gold_dois - pred_dois
    total = len(common_dois)

    print(f"\n=== DOI Coverage ===")
    print(f"  Common DOIs (can compare): {total}")
    print(f"  Prediction-only DOIs (no gold): {len(pred_only)}")
    print(f"  Gold-only DOIs (no prediction): {len(gold_only)}")

    shared_kwargs = dict(
        common_dois=common_dois, gold=gold, preds=preds,
        total_compared=total, parse_errors=parse_errors,
        pred_file=args.predictions, gold_file=args.gold,
        threshold=args.threshold,
        pred_count=len(pred_dois), gold_count=len(gold_dois),
        pred_only=pred_only, gold_only=gold_only,
    )

    print(f"\n{'=' * 70}")
    print("Running STRICT comparison (exact match after normalization)...")
    print(f"{'=' * 70}")
    s_errors, s_summary, s_perfect = run_comparison(
        common_dois, preds, gold, use_fuzzy=False, threshold=0)
    write_outputs(
        os.path.join(args.output_dir, "strict"),
        s_errors, s_summary, perfect_count=s_perfect,
        mode_label="STRICT", **shared_kwargs)

    print(f"\n{'=' * 70}")
    print(f"Running FUZZY comparison (threshold={args.threshold})...")
    print(f"{'=' * 70}")
    f_errors, f_summary, f_perfect = run_comparison(
        common_dois, preds, gold, use_fuzzy=True, threshold=args.threshold)
    write_outputs(
        os.path.join(args.output_dir, "fuzzy"),
        f_errors, f_summary, perfect_count=f_perfect,
        mode_label=f"FUZZY (threshold={args.threshold})", **shared_kwargs)

    s_err_dois = total - s_perfect
    f_err_dois = total - f_perfect
    print(f"\n{'=' * 70}")
    print("STRICT vs FUZZY COMPARISON")
    print(f"{'=' * 70}")
    print(f"  Strict: {s_perfect} perfect ({100 * s_perfect / total:.1f}%), "
          f"{len(s_errors)} total errors")
    print(f"  Fuzzy:  {f_perfect} perfect ({100 * f_perfect / total:.1f}%), "
          f"{len(f_errors)} total errors")
    print(f"  DOIs resolved by fuzzy matching: {s_err_dois - f_err_dois}")
    print(f"  Errors resolved by fuzzy matching: {len(s_errors) - len(f_errors)}")
    print("\nDone!")


if __name__ == "__main__":
    main()
