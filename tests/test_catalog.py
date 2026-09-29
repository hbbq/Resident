import unittest

from resident.catalog import ROOT, render


class CatalogTests(unittest.TestCase):
    def test_checked_in_catalog_matches_runtime_inventory(self) -> None:
        checked_in = (ROOT / "CAPABILITIES.md").read_text(encoding="utf-8")
        self.assertEqual(checked_in, render(),
                         "CAPABILITIES.md is stale; run python -m resident.catalog")
