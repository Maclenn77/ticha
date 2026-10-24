from pathlib import Path


def test_osx_arm64_lock_file_does_not_require_ticha_from_pypi():
    lock_file = (
        Path(__file__).resolve().parent.parent / "environment.osx-arm64.lock.yml"
    )
    assert "ticha==" not in lock_file.read_text(encoding="utf-8")
