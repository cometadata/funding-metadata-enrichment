from funding_statement_extractor_cascade.io.checkpointing import CheckpointRepository, get_file_hash


def test_hash_is_deterministic():
    assert get_file_hash("a.md") == get_file_hash("a.md")
    assert get_file_hash("a.md") != get_file_hash("b.md")


def test_repo_records_and_round_trips(tmp_path):
    p = tmp_path / "ck.json"
    repo = CheckpointRepository(p)
    repo.record("h1", {"path": "a.md", "status": "ok"})
    repo.save()

    fresh = CheckpointRepository(p)
    data = fresh.load(resume=True)
    assert data["processed_files"]["h1"]["status"] == "ok"
    assert fresh.is_processed("h1")
    assert not fresh.is_processed("nope")


def test_save_is_atomic(tmp_path):
    p = tmp_path / "ck.json"
    repo = CheckpointRepository(p)
    repo.save()
    assert not (tmp_path / "ck.json.tmp").exists()
    assert p.exists()
