from __future__ import annotations

import pathlib
import subprocess
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the Streamlit app entrypoint for OPTIMEO."""
    args = list(argv) if argv is not None else sys.argv[1:]
    app_dir = pathlib.Path(__file__).resolve().parent / "app"
    home_py = app_dir / "Home.py"

    if not home_py.exists():
        raise FileNotFoundError(f"Could not find app entrypoint at {home_py}")

    has_file_watcher_arg = any(
        arg == "--server.fileWatcherType" or arg.startswith("--server.fileWatcherType=")
        for arg in args
    )
    # Work around Streamlit watcher crashes with torch.classes path inspection.
    default_streamlit_args = [] if has_file_watcher_arg else ["--server.fileWatcherType=none"]

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(home_py),
        *default_streamlit_args,
        *args,
    ]
    return subprocess.call(cmd, cwd=str(app_dir))


if __name__ == "__main__":
    raise SystemExit(main())
