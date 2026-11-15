"""Path B worker: `python -m storyreel.render.worker <spec.json>`.

Runs in its own process under the Job Object memory cap (jobs.limits.spawn). Composites
the chunk frame by frame and pipes rgb24 into an ffmpeg encoder. Prints ffmpeg-style
`frame=N` / `progress=...` lines so jobs.runner's heartbeat works unchanged.
Exit codes: 0 ok, 1 error (message on stderr), 3 out of memory.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any


def run(spec: dict[str, Any]) -> None:
    from storyreel.render.compositor import Compositor

    s = spec["settings"]
    w, h, n = int(s["width"]), int(s["height"]), int(spec["frames"])
    ffmpeg = Path(spec["ffmpeg"])
    comp = Compositor(spec, ffmpeg)
    argv = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{s['fps_num']}/{s['fps_den']}",
        "-i",
        "pipe:0",
        *spec["encoder_args"],
        "-frames:v",
        str(n),
        "-f",
        "matroska",
        spec["out"],
    ]
    with tempfile.TemporaryFile() as errf:
        enc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errf)
        assert enc.stdin is not None
        try:
            for f in range(n):
                enc.stdin.write(comp.render(f).tobytes())
                if f % 10 == 9 or f == n - 1:
                    print(f"frame={f + 1}\nprogress=continue", flush=True)
            enc.stdin.close()
            rc = enc.wait()
        except BaseException:
            enc.kill()
            enc.wait()
            raise
        finally:
            comp.close()
        if rc != 0:
            errf.seek(0)
            raise RuntimeError(
                f"encoder failed ({rc}): {errf.read().decode('utf-8', 'replace')[-800:]}"
            )
    print("progress=end", flush=True)


def main() -> int:
    try:
        spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        run(spec)
    except MemoryError:
        print("MemoryError: compositor ran out of memory", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001  # process boundary: report any error to the parent
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
