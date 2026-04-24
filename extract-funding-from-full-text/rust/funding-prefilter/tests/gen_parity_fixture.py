#!/usr/bin/env python3
"""Generate the parity fixture used by tests/parity.rs.

This script reconstructs the Python gate inline (intentionally independent of the
`funding_statement_extractor` package so it runs with only `yaml` available) and
uses it to compute the expected pass/fail for each curated paragraph. The output
is written to `tests/fixtures/parity_inputs.json` as a JSON array of
`{paragraph, expected, category}` records.

Run (from the repo root):

    .venv/bin/python3 rust/funding-prefilter/tests/gen_parity_fixture.py
"""

import json
import re
import sys
from pathlib import Path

import yaml

# Reconstructed Python gate --------------------------------------------------

PREFILTER = re.compile(
    r"\b(fund|grant|support|acknowledg|award|sponsor|thank|scholarship|fellowship|financ|grate|gratitude|foundation)\w*\b"
    r"|\b(?:NSF|NSFC|NIH|NASA|ESA|CNES|DOE|ERC|EPSRC|DFG|JSPS|MCIN|AEI|FAPESP|CNPq|JPL|CSIC|CONICET|CONACYT|RFBR|HFSP|JST|MEXT|KAKENHI)\b"
    r"|\bin\s+(?:the\s+)?(?:framework|scope|context)\s+of\b"
    r"|\bis\s+part\s+of\s+the\s+(?:project|research|R\+D\+i)\b"
    r"|\bcarried\s+out\s+(?:within|as|in|during)\b"
    r"|\b(?:state\s+assignment|госзадания)\b",
    re.IGNORECASE,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
YAML_PATH = REPO_ROOT / "funding_statement_extractor/configs/patterns/funding_patterns.yaml"

data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
POSITIVES = [re.compile(p, re.IGNORECASE) for p in data["patterns"]]
NEGATIVES = [re.compile(p, re.IGNORECASE) for p in data["negative_patterns"]]


def gate(para: str) -> bool:
    if not PREFILTER.search(para):
        return False
    if not any(p.search(para) for p in POSITIVES):
        return False
    if any(n.search(para) for n in NEGATIVES):
        return False
    return True


# Curated samples ------------------------------------------------------------

POSITIVES_SAMPLES = [
    "This work was supported by the NSF under grant number AST-1234567.",
    "We acknowledge funding from the European Research Council (ERC) under grant 101002811.",
    "This research was funded by JSPS KAKENHI Grant Number JP18K12345.",
    "J.D. acknowledges financial support from the Simons Foundation.",
    "Funding: This work was supported by NASA grant 80NSSC19K0001.",
    "The authors thank the DFG for funding under project number 12345678.",
    "This study received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 000000.",
    "Work supported in part by DOE grant DE-FG02-91ER40628.",
    "This research was supported by a fellowship from the Alfred P. Sloan Foundation.",
    "We are grateful for the hospitality of the Aspen Center for Physics, which is supported by NSF grant PHY-2210452.",
    "The author gratefully acknowledges the support of the EPSRC through grant EP/X000000/1.",
    "This work is part of the research programme supported by CNPq.",
    "Scholarship recipient under the Chinese Scholarship Council (CSC).",
    "Funding provided by the Max Planck Society.",
    "Carried out within the framework of the state assignment of the Russian Academy of Sciences.",
    "This research was supported in part by CONACYT fellowship 987654.",
    "Financially supported by FAPESP grant 2020/00000-0.",
    "Award #PHY-1912345 from the National Science Foundation.",
    "This project has received funding from AEI/MCIN under grant PID2020-117123RB-I00.",
    "We thank CSIC for providing computational facilities and support.",
]

NEGATIVES_SAMPLES = [
    "We present a new method for analyzing galaxy clusters.",
    "The quantum harmonic oscillator is a model studied in undergraduate physics.",
    "Figure 3 shows the distribution of masses as a function of redshift.",
    "Our numerical simulations were carried out using the MPI framework.",
    "The ratio of these quantities is given in Eq. (12).",
    "In this paper we propose a novel architecture.",
    "These results are consistent with theoretical predictions.",
    "The observations were taken on three consecutive nights.",
    "We used Python and numpy for the numerical work.",
    "Data is available from the authors on reasonable request.",
    "The temperature profile is shown in Figure 5.",
    "A review of the literature is presented in the next section.",
    "We adopt the cosmology from Planck 2018 results.",
    "The mean value was computed as the arithmetic average.",
    "The dataset consists of 10,000 simulated trajectories.",
    "Our model predicts a correlation between X and Y.",
    "We refer the reader to Section 3 for the derivation.",
    "This is shown on the right panel of Figure 2.",
    "Thanks to all the co-authors for their feedback.",
    "These predictions are supported by our experimental measurements.",
]

HARD_NEGATIVES_SAMPLES = [
    "The Higgs mechanism is supported by theoretical arguments and experimental data.",
    "The proof is supported by Lemma 3.2.",
    "Granted by the institutional review board under protocol 12345.",
    "The claim is supported by the equation above.",
    "Work done at CERN.",
    "This result is supported by several observations and simulations.",
    "Our approach is supported by recent work.",
    "The conclusion is supported by both theoretical and numerical evidence.",
    "Hospitality of the host institution is gratefully acknowledged.",
    "The framework of relativity is well established.",
]


def build_records():
    records = []
    for text in POSITIVES_SAMPLES:
        records.append({
            "paragraph": text,
            "expected": gate(text),
            "category": "positive",
        })
    for text in NEGATIVES_SAMPLES:
        records.append({
            "paragraph": text,
            "expected": gate(text),
            "category": "negative",
        })
    for text in HARD_NEGATIVES_SAMPLES:
        records.append({
            "paragraph": text,
            "expected": gate(text),
            "category": "hard_negative",
        })
    return records


def main() -> int:
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "parity_inputs.json"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)

    records = build_records()
    fixture_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    total = len(records)
    n_true = sum(1 for r in records if r["expected"])
    n_false = total - n_true
    by_cat = {}
    for r in records:
        by_cat.setdefault(r["category"], {"true": 0, "false": 0})
        by_cat[r["category"]]["true" if r["expected"] else "false"] += 1

    print(f"wrote {fixture_path}")
    print(f"  total={total}  expected_true={n_true}  expected_false={n_false}")
    for cat, counts in by_cat.items():
        print(f"  {cat}: true={counts['true']}  false={counts['false']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
