import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funding_extractor.providers.base import BaseProvider
from funding_extractor.entities.models import Award


def _ext(cls, text, attrs=None):
    """Helper to create a mock langextract Extraction object."""
    return SimpleNamespace(extraction_class=cls, extraction_text=text, attributes=attrs)


def test_single_funder_single_award():
    extractions = [
        _ext("funder_name", "NSF"),
        _ext("award_ids", "123", {"funder_name": "NSF"}),
    ]
    result = BaseProvider._convert_extractions_to_result(extractions, "Funded by NSF grant 123")
    assert len(result.funders) == 1
    f = result.funders[0]
    assert f.funder_name == "NSF"
    assert len(f.awards) == 1
    assert f.awards[0].award_ids == ["123"]
    assert f.awards[0].funding_scheme == []


def test_single_funder_multiple_ids_same_award():
    extractions = [
        _ext("funder_name", "NSF"),
        _ext("award_ids", "123", {"funder_name": "NSF"}),
        _ext("award_ids", "456", {"funder_name": "NSF"}),
    ]
    result = BaseProvider._convert_extractions_to_result(extractions, "stmt")
    f = result.funders[0]
    assert len(f.awards) == 1
    assert f.awards[0].award_ids == ["123", "456"]


def test_funder_with_scheme_groups_into_award():
    extractions = [
        _ext("funder_name", "USDA"),
        _ext("funding_scheme", "Multistate Research Project", {"funder_name": "USDA"}),
        _ext("award_ids", "PEN04623", {"funder_name": "USDA", "funding_scheme": "Multistate Research Project"}),
        _ext("award_ids", "1013257", {"funder_name": "USDA", "funding_scheme": "Multistate Research Project"}),
    ]
    result = BaseProvider._convert_extractions_to_result(extractions, "stmt")
    f = result.funders[0]
    assert f.funder_name == "USDA"
    assert len(f.awards) == 1
    assert f.awards[0].funding_scheme == ["Multistate Research Project"]
    assert set(f.awards[0].award_ids) == {"PEN04623", "1013257"}


def test_funder_with_multiple_distinct_schemes():
    extractions = [
        _ext("funder_name", "DOE"),
        _ext("funding_scheme", "Program A", {"funder_name": "DOE"}),
        _ext("award_ids", "100", {"funder_name": "DOE", "funding_scheme": "Program A"}),
        _ext("funding_scheme", "Program B", {"funder_name": "DOE"}),
        _ext("award_ids", "200", {"funder_name": "DOE", "funding_scheme": "Program B"}),
    ]
    result = BaseProvider._convert_extractions_to_result(extractions, "stmt")
    f = result.funders[0]
    assert len(f.awards) == 2
    schemes = {tuple(a.funding_scheme) for a in f.awards}
    assert schemes == {("Program A",), ("Program B",)}


def test_award_title_attached_to_correct_award():
    extractions = [
        _ext("funder_name", "ERC"),
        _ext("award_ids", "101053661", {"funder_name": "ERC"}),
        _ext("award_title", "COMP-O-CELL", {"funder_name": "ERC"}),
    ]
    result = BaseProvider._convert_extractions_to_result(extractions, "stmt")
    f = result.funders[0]
    assert len(f.awards) == 1
    assert f.awards[0].award_ids == ["101053661"]
    assert f.awards[0].award_title == ["COMP-O-CELL"]


def test_funder_name_not_in_pass1_created_by_award_ids():
    """If award_ids references a funder not seen in funder_name extractions, create it."""
    extractions = [
        _ext("award_ids", "789", {"funder_name": "Mystery Fund"}),
    ]
    result = BaseProvider._convert_extractions_to_result(extractions, "stmt")
    assert len(result.funders) == 1
    assert result.funders[0].funder_name == "Mystery Fund"
    assert result.funders[0].awards[0].award_ids == ["789"]


def test_multiple_funders():
    extractions = [
        _ext("funder_name", "NSF"),
        _ext("funder_name", "NIH"),
        _ext("award_ids", "A1", {"funder_name": "NSF"}),
        _ext("award_ids", "B1", {"funder_name": "NIH"}),
    ]
    result = BaseProvider._convert_extractions_to_result(extractions, "stmt")
    assert len(result.funders) == 2
    names = {f.funder_name for f in result.funders}
    assert names == {"NSF", "NIH"}


def test_funder_with_no_awards_gets_empty_award():
    extractions = [
        _ext("funder_name", "SURF"),
    ]
    result = BaseProvider._convert_extractions_to_result(extractions, "stmt")
    f = result.funders[0]
    assert f.funder_name == "SURF"
    assert len(f.awards) == 1
    assert f.awards[0].award_ids == []
    assert f.awards[0].funding_scheme == []


def test_dedup_award_ids():
    """Same award_id mentioned twice should not be duplicated."""
    extractions = [
        _ext("funder_name", "NSF"),
        _ext("award_ids", "123", {"funder_name": "NSF"}),
        _ext("award_ids", "123", {"funder_name": "NSF"}),
    ]
    result = BaseProvider._convert_extractions_to_result(extractions, "stmt")
    assert result.funders[0].awards[0].award_ids == ["123"]


def test_empty_extractions():
    result = BaseProvider._convert_extractions_to_result([], "stmt")
    assert result.funders == []
    assert result.statement == "stmt"


class _StubProvider(BaseProvider):
    """Minimal concrete provider for testing _execute_extract."""
    @property
    def provider(self):
        return "stub"
    def build_extract_params(self, statement, prompt, examples):
        return {}


def test_execute_extract_returns_empty_on_value_error():
    """ValueError from resolver (e.g. model returns list instead of string) yields empty result."""
    with patch("funding_extractor.providers.base.lx") as mock_lx:
        mock_lx.extract.side_effect = ValueError(
            "Extraction text must be a string, integer, or float."
        )
        provider = _StubProvider(model_id=None, model_url=None, api_key=None, timeout=30)
        params = {"text_or_documents": "stmt", "model": MagicMock()}
        result = provider._execute_extract(params, "stmt")
        assert result.funders == []
        assert result.statement == "stmt"


def test_execute_extract_returns_empty_on_timeout():
    """TimeoutError yields empty result (pre-existing behavior)."""
    import threading

    with patch("funding_extractor.providers.base.lx") as mock_lx:
        event = threading.Event()
        def block_forever(**kwargs):
            event.wait()
        mock_lx.extract.side_effect = block_forever

        provider = _StubProvider(model_id=None, model_url=None, api_key=None, timeout=0.01)
        params = {"text_or_documents": "stmt", "model": MagicMock()}
        result = provider._execute_extract(params, "stmt")
        event.set()
        assert result.funders == []
        assert result.statement == "stmt"
