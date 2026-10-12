"""Supported standalone worker entry; source execution also uses the broker."""
from pathlib import Path
import sys
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localocr.runtime_entry import main
if __name__ == "__main__":
    raise SystemExit(main())
