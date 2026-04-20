# -*- coding: utf-8 -*-

from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent.parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def out_path(filename: str) -> Path:
    return OUT_DIR / filename
