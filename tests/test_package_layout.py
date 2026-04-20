from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
IMPORT_PATTERN = re.compile(r"(?:from|import)\s+code(?:\.|\b)")


class PackageLayoutTests(unittest.TestCase):
    def test_legacy_code_package_directory_is_gone(self):
        self.assertFalse((ROOT_DIR / "code").exists())

    def test_repository_sources_do_not_import_legacy_code_package(self):
        allowed_suffixes = {".ipynb", ".md", ".py", ".toml"}
        search_roots = [
            ROOT_DIR / "notebooks",
            ROOT_DIR / "tests",
            ROOT_DIR / "xai_book",
            ROOT_DIR / "README.md",
            ROOT_DIR / "pyproject.toml",
        ]

        offenders: list[str] = []
        for path in search_roots:
            if path.is_file():
                candidates = [path]
            else:
                candidates = sorted(
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file() and candidate.suffix in allowed_suffixes
                )

            for candidate in candidates:
                if IMPORT_PATTERN.search(candidate.read_text(encoding="utf-8", errors="ignore")):
                    offenders.append(str(candidate.relative_to(ROOT_DIR)))

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
