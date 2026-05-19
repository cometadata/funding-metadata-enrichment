
import pytest

from funding_statement_extractor_cascade.config.settings import (
    ApplicationConfig,
    CascadeRuntimeSettings,
    InputSettings,
    OutputSettings,
    ProcessingSettings,
    RuntimeSettings,
)
from funding_statement_extractor_cascade.exceptions import ConfigurationError


def _make_app(tmp_path, **overrides):
    src = tmp_path / "in.md"
    src.write_text("x")
    return ApplicationConfig(
        input=InputSettings(path=src),
        output=OutputSettings(output_path=tmp_path / "out.json",
                              checkpoint_path=tmp_path / "out.json.checkpoint"),
        cascade=CascadeRuntimeSettings(),
        processing=ProcessingSettings(),
        runtime=RuntimeSettings(),
    )


def test_default_runtime_workers_is_none(tmp_path):
    cfg = _make_app(tmp_path)
    assert cfg.runtime.workers is None


def test_default_cascade_settings_align_with_cascade_config(tmp_path):
    cfg = _make_app(tmp_path)
    assert cfg.cascade.skip_cleanup is False
    assert cfg.cascade.dtype == "bf16"
    assert cfg.cascade.chunk_keep_threshold == 0.4


def test_validate_passes_with_existing_input(tmp_path):
    cfg = _make_app(tmp_path)
    cfg.validate()


def test_validate_rejects_missing_input(tmp_path):
    src = tmp_path / "missing.md"  # not created
    cfg = ApplicationConfig(
        input=InputSettings(path=src),
        output=OutputSettings(output_path=tmp_path / "o.json",
                              checkpoint_path=tmp_path / "c.json"),
        cascade=CascadeRuntimeSettings(),
        processing=ProcessingSettings(),
        runtime=RuntimeSettings(),
    )
    with pytest.raises(ConfigurationError):
        cfg.validate()
