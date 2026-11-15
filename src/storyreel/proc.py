"""Run external tools with a timeout and typed errors. No shell, argv lists only."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from storyreel.errors import ToolFailedError, ToolNotFoundError, ToolTimeoutError

STDERR_TAIL_LINES = 50


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str


def tail(text: str, lines: int = STDERR_TAIL_LINES) -> str:
    return "\n".join(text.splitlines()[-lines:])


def run(
    argv: Sequence[str | Path],
    *,
    timeout_s: float,
    check: bool = True,
    stdin_data: str | None = None,
) -> RunResult:
    """Run argv, capture text output, enforce timeout.

    Raises ToolNotFoundError, ToolTimeoutError, or (if check) ToolFailedError.
    """
    args = [str(a) for a in argv]
    try:
        cp = subprocess.run(
            args,
            input=stdin_data,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ToolNotFoundError(f"Program not found: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        # subprocess.run kills the child before re-raising TimeoutExpired.
        raise ToolTimeoutError(
            f"{Path(args[0]).name} did not finish within {timeout_s:.0f} s"
        ) from exc
    if check and cp.returncode != 0:
        raise ToolFailedError(
            f"{Path(args[0]).name} exited with code {cp.returncode}",
            returncode=cp.returncode,
            stderr_tail=tail(cp.stderr),
        )
    return RunResult(cp.returncode, cp.stdout, cp.stderr)
