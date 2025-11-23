import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funding_extractor.processing.markdown_healer import heal_markdown


def test_heal_markdown_collapses_fragment_runs():
    raw = (
        "After\n\n"
        "results\n\n"
        "analyzing\n\n"
        "from Multi-\n"
        "recognition\n"
        "Transformer support\n"
    )

    healed = heal_markdown(raw)
    assert "After results analyzing" in healed
    assert "Multirecognition Transformer support" in healed
    assert "After\n\nresults" not in healed


def test_heal_markdown_cleans_vertical_runs():
    raw = (
        "Heading\n\n"
        "8\n"
        "1\n"
        "0\n"
        "2\n"
        "\n"
        "n\n"
        "u\n"
        "J\n\n"
        "Body paragraph about funding."
    )

    healed = heal_markdown(raw)
    assert "8\n1\n0\n2" not in healed
    assert "Body paragraph about funding." in healed


def test_heal_markdown_leaves_regular_paragraphs():
    paragraph = (
        "This work was supported by the National Science Foundation under grant 12345. "
        "Additional support was provided by NASA."
    )
    healed = heal_markdown(paragraph)
    assert healed == paragraph


def test_heal_markdown_reassembles_funding_section():
    raw = (
        "Funding\n\n"
        "The work was\n"
        "000000D730321P5Q0002, Grant No. 70-2021-00145 02.11.2021).\n\n"
        "supported by the Analytical center under\n\n"
        "the RF Government\n\n"
        "(subsidy agreement\n\n"
        "Author contributions statement\n"
        "All authors contributed.\n"
    )

    healed = heal_markdown(raw)
    assert "Funding" in healed
    assert "The work was 000000D730321P5Q0002, Grant No. 70-2021-00145 02.11.2021). supported by the Analytical center under the RF Government (subsidy agreement" in healed
    assert "Author contributions statement" in healed
    assert "All authors contributed." in healed


def test_heal_markdown_stops_at_reference_headings():
    raw = (
        "Table 5: Rand index scores, real datasets\n\n"
        "and real datasets. Our experimental results showed that CAST performed very well against its competitors over all the datasets.\n\n"
        "6 ACKNOWLEDGMENTS\n"
        "This research is supported by Hong Kong Research Grants Council GRF HKU 17254016.\n\n"
        "REFERENCES\n"
        "[1] Charles J Alpert and So-Zen Yao. 1995. Spectral partitioning: the more eigenvectors, the better.\n"
    )

    healed = heal_markdown(raw)
    assert "This research is supported by Hong Kong Research Grants Council GRF HKU 17254016." in healed
    assert "\nREFERENCES" in healed
    assert "GRF HKU 17254016.\n\nREFERENCES" in healed
