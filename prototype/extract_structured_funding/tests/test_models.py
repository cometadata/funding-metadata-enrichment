"""Tests for data models."""

import pytest

from funding_extractor.models import (
    FundingEntity,
    FundingExtractionResult,
    Grant,
    ProcessingStats,
)


class TestGrant:
    """Tests for Grant model."""

    def test_grant_creation(self) -> None:
        grant = Grant(grant_id="R01-123456")
        assert grant.grant_id == "R01-123456"

    def test_grant_with_different_id(self) -> None:
        grant = Grant(grant_id="DMS-1234567")
        assert grant.grant_id == "DMS-1234567"


class TestFundingEntity:
    """Tests for FundingEntity model."""

    def test_basic_entity_creation(self) -> None:
        entity = FundingEntity(
            funder="National Science Foundation",
            grants=[Grant(grant_id="DMS-1234567")],
            extraction_texts=["NSF grant DMS-1234567"],
        )
        assert entity.funder == "National Science Foundation"
        assert len(entity.grants) == 1
        assert entity.grants[0].grant_id == "DMS-1234567"
        assert "NSF grant DMS-1234567" in entity.extraction_texts

    def test_entity_with_multiple_grants(self) -> None:
        entity = FundingEntity(funder="NIH")
        entity.add_grant("R01-123456")
        entity.add_grant("T32-789012")
        entity.add_grant("R01-345678")

        assert entity.funder == "NIH"
        assert len(entity.grants) == 3
        assert entity.grants[0].grant_id == "R01-123456"
        assert entity.grants[1].grant_id == "T32-789012"
        assert entity.grants[2].grant_id == "R01-345678"

    def test_duplicate_grant_handling(self) -> None:
        entity = FundingEntity(funder="NSF")
        entity.add_grant("DMS-1234567")
        entity.add_grant("DMS-1234567")

        # Should not duplicate
        assert len(entity.grants) == 1
        assert entity.grants[0].grant_id == "DMS-1234567"


class TestFundingExtractionResult:
    """Tests for FundingExtractionResult model."""

    def test_result_creation(self) -> None:
        nsf_entity = FundingEntity(funder="NSF")
        nsf_entity.add_grant("1234567")

        nih_entity = FundingEntity(funder="NIH")
        nih_entity.add_grant("R01-123456")
        nih_entity.add_grant("T32-789012")

        entities = [nsf_entity, nih_entity]

        result = FundingExtractionResult(
            doi="10.1234/example.2024",
            funding_statement=(
                "Supported by NSF 1234567, NIH R01-123456 and T32-789012."
            ),
            entities=entities,
        )

        assert result.doi == "10.1234/example.2024"
        assert (
            result.funding_statement
            == "Supported by NSF 1234567, NIH R01-123456 and T32-789012."
        )
        assert len(result.entities) == 2
        assert result.entities[0].funder == "NSF"
        assert result.entities[1].funder == "NIH"
        assert len(result.entities[1].grants) == 2

    def test_result_with_empty_entities(self) -> None:
        result = FundingExtractionResult(
            doi="10.1234/empty.2024",
            funding_statement="No specific funding.",
            entities=[],
        )
        assert result.doi == "10.1234/empty.2024"
        assert len(result.entities) == 0


class TestProcessingStats:
    """Tests for ProcessingStats model."""

    def test_stats_creation(self) -> None:
        stats = ProcessingStats(
            total_documents=100,
            successful=95,
            failed=5,
            total_entities=250,
        )
        assert stats.total_documents == 100
        assert stats.successful == 95
        assert stats.failed == 5
        assert stats.total_entities == 250

    @pytest.mark.parametrize(
        "total,success,fail,entities",
        [
            (0, 0, 0, 0),
            (1, 1, 0, 3),
            (10, 8, 2, 25),
            (1000, 999, 1, 5000),
        ],
    )
    def test_various_stats(
        self, total: int, success: int, fail: int, entities: int
    ) -> None:
        stats = ProcessingStats(
            total_documents=total,
            successful=success,
            failed=fail,
            total_entities=entities,
        )
        assert stats.total_documents == total
        assert stats.successful == success
        assert stats.failed == fail
        assert stats.total_entities == entities
        assert stats.successful + stats.failed <= stats.total_documents
