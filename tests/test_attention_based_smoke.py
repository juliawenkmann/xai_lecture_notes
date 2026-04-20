from __future__ import annotations

import unittest


class AttentionBasedSmokeTests(unittest.TestCase):
    def test_attention_module_imports(self):
        import xai_book.attention_based  # noqa: F401


if __name__ == "__main__":
    unittest.main()
