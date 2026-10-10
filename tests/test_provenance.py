from pathlib import Path
import re

from flowoe_execution.provenance import source_revision, utc_now, workflow_revision


def test_source_revision_records_full_commit_in_repository():
    root = Path(__file__).resolve().parents[1]
    revision = source_revision(root)
    assert revision is not None
    assert re.fullmatch(r"[0-9a-f]{40}", revision)


def test_source_revision_is_optional_outside_git_checkout(tmp_path):
    assert source_revision(tmp_path) is None


def test_utc_timestamp_is_timezone_aware():
    value = utc_now()
    assert value.endswith("+00:00")


def test_workflow_revision_accepts_full_commit_sha(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    assert workflow_revision() == "a" * 40


def test_workflow_revision_rejects_invalid_sha(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "not-a-commit")
    assert workflow_revision() is None
