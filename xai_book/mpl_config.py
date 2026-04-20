from __future__ import annotations

import os
import tempfile
from pathlib import Path

fallback_mpl_dir = Path(tempfile.gettempdir()) / "xai_book_mpl"
default_mpl_dir = Path.home() / ".matplotlib"

if "MPLCONFIGDIR" not in os.environ and not os.access(default_mpl_dir, os.W_OK):
    fallback_mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(fallback_mpl_dir)
