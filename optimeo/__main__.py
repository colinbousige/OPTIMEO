from __future__ import annotations

import pathlib
import subprocess
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the Streamlit app entrypoint for OPTIMEO."""
    args = list(argv) if argv is not None else sys.argv[1:]
    home_py = pathlib.Path(__file__).resolve().parents[1] / "Home.py"

    if not home_py.exists():
        raise FileNotFoundError(f"Could not find app entrypoint at {home_py}")

    cmd = [sys.executable, "-m", "streamlit", "run", str(home_py), *args]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
