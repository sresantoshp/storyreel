from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from storyreel import cli, doctor, logs, paths
from storyreel.errors import ExitCode


@pytest.fixture
def no_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.delenv("STORYREEL_FFMPEG", raising=False)
    monkeypatch.delenv("STORYREEL_FFPROBE", raising=False)
    monkeypatch.setattr(paths, "bin_dir", lambda: empty)
    monkeypatch.setattr(paths, "data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr(paths, "log_dir", lambda: tmp_path / "logs")
    return tmp_path


def test_missing_ffmpeg_is_fail_with_fix(no_ffmpeg: Path) -> None:
    checks = {c.name: c for c in doctor.run_checks(test_hw=False, env={})}
    assert checks["ffmpeg"].status is doctor.Status.FAIL
    assert checks["ffmpeg"].fix
    assert checks["SQLite WAL"].status is doctor.Status.OK
    assert checks["Key ANTHROPIC_API_KEY"].status is doctor.Status.INFO
    report = doctor.format_report(list(checks.values()))
    assert "fix:" in report and "FAIL=" in report


def test_api_keys_reported_as_presence_only(no_ffmpeg: Path) -> None:
    checks = doctor.run_checks(test_hw=False, env={"ANTHROPIC_API_KEY": "sk-secret-value"})
    text = doctor.format_report(checks) + json.dumps([c.detail for c in checks])
    assert "sk-secret-value" not in text


def test_cli_doctor_exit_code_3_when_ffmpeg_missing(
    no_ffmpeg: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["doctor", "--json", "--no-hw-test"])
    assert rc == ExitCode.ENV_CHECK_FAILED == 3
    parsed = json.loads(capsys.readouterr().out)
    assert any(c["name"] == "ffmpeg" and c["status"] == "FAIL" for c in parsed)


def test_cli_logs_json_lines(no_ffmpeg: Path) -> None:
    cli.main(["doctor", "--no-hw-test"])
    log_file = no_ffmpeg / "logs" / "storyreel.jsonl"
    lines = [json.loads(ln) for ln in log_file.read_text(encoding="utf-8").splitlines()]
    assert {"command start", "command end"} <= {ln["msg"] for ln in lines}
    assert all("ts" in ln and "level" in ln for ln in lines)


def test_unknown_command_is_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["no-such-command"])
    assert exc.value.code == ExitCode.USAGE


def test_json_formatter_handles_extra_fields(tmp_path: Path) -> None:
    path = logs.configure(tmp_path)
    logging.getLogger("storyreel.test").info("hi", extra={"job": 7, "path": Path("x")})
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["job"] == 7 and rec["msg"] == "hi"
