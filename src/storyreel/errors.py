"""Typed errors and process exit codes.

Every error carries a plain-English message for the user and an optional one-line fix.
Stack traces go to the log file, never to the user.
"""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    ERROR = 1  # unexpected internal error (bug); details in the log
    USAGE = 2  # bad command-line arguments (argparse uses 2 as well)
    ENV_CHECK_FAILED = 3  # doctor found a FAIL, or a required tool is missing
    RENDER_FAILED = 4
    RESOURCE_EXHAUSTED = 5  # memory cap, disk full, budget cap
    INTERRUPTED = 130


class StoryReelError(Exception):
    """Base class. `message` is shown to the user; `fix` is a suggested next action."""

    exit_code: ExitCode = ExitCode.ERROR

    def __init__(self, message: str, *, fix: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.fix = fix


class ToolNotFoundError(StoryReelError):
    exit_code = ExitCode.ENV_CHECK_FAILED


class ToolTimeoutError(StoryReelError):
    pass


class ToolFailedError(StoryReelError):
    def __init__(
        self, message: str, *, returncode: int, stderr_tail: str, fix: str | None = None
    ) -> None:
        super().__init__(message, fix=fix)
        self.returncode = returncode
        self.stderr_tail = stderr_tail
