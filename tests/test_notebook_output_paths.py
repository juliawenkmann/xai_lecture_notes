from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
FORBIDDEN_LITERAL_PATTERN = re.compile(r"""['"](results|figures|images)/""")


class NotebookOutputPathTests(unittest.TestCase):
    def test_notebook_code_cells_do_not_use_legacy_output_directories(self):
        offenders: list[str] = []

        for path in sorted(NOTEBOOKS_DIR.rglob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                if FORBIDDEN_LITERAL_PATTERN.search(source):
                    offenders.append(f"{path.relative_to(ROOT_DIR)}#cell-{index}")

        self.assertEqual(offenders, [])

    def test_notebook_python_helpers_do_not_use_legacy_output_directories(self):
        offenders: list[str] = []

        for path in sorted(NOTEBOOKS_DIR.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            if FORBIDDEN_LITERAL_PATTERN.search(source):
                offenders.append(str(path.relative_to(ROOT_DIR)))

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
