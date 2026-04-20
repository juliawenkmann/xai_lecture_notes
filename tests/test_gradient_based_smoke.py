from __future__ import annotations

import unittest

from xai_book.gradient_based import default_gradcam_image_path


class GradientBasedSmokeTests(unittest.TestCase):
    def test_default_gradcam_image_exists(self):
        self.assertTrue(default_gradcam_image_path().exists())


if __name__ == "__main__":
    unittest.main()
