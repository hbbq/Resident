import unittest
import ast
from unittest.mock import patch

from resident import catalog
from resident.__main__ import _select_capabilities
from resident.catalog import ROOT, render


class CatalogTests(unittest.TestCase):
    def test_checked_in_catalog_matches_runtime_inventory(self) -> None:
        checked_in = (ROOT / "CAPABILITIES.md").read_text(encoding="utf-8")
        self.assertEqual(checked_in, render(),
                         "CAPABILITIES.md is stale; run python -m resident.catalog")

    def _render_with_source_change(self, filename: str, old: str, new: str) -> str:
        original = catalog._tree
        source = (ROOT / "resident" / filename).read_text(encoding="utf-8")
        self.assertIn(old, source)
        changed = ast.parse(source.replace(old, new, 1))

        def tree(name: str) -> ast.Module:
            return changed if name == filename else original(name)

        with patch.object(catalog, "_tree", side_effect=tree):
            return render()

    def test_dynamic_capability_patterns_must_be_accounted_for(self) -> None:
        changes = (
            ("display.py", 'name=f"{display_id}_show_text"',
             'name=f"{display_id}_clear_text"'),
            ("external_app.py", "name=operation.name", "name=operation.alias"),
            ("realm.py", "for name, description, schema, route in specs",
             "for name, description, schema, route in extra_specs"),
        )
        for filename, old, new in changes:
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "Uncataloged dynamic capability"):
                    self._render_with_source_change(filename, old, new)

    def test_new_dynamic_capability_calls_cannot_be_silently_skipped(self) -> None:
        for filename in ("display.py", "external_app.py", "realm.py"):
            with self.subTest(filename=filename):
                extra = "\nCapability(new_connector, 'Extra', new_name, 'Extra', {}, None)\n"
                with self.assertRaisesRegex(ValueError, "Uncataloged dynamic capability"):
                    self._render_with_source_change(
                        filename, "from __future__ import annotations",
                        "from __future__ import annotations" + extra)

    def test_new_realm_spec_makes_checked_in_catalog_stale(self) -> None:
        changed = self._render_with_source_change(
            "realm.py", '("realm_read",', '("realm_new_action",')
        self.assertIn("`realm_new_action`", changed)
        self.assertNotEqual((ROOT / "CAPABILITIES.md").read_text(encoding="utf-8"), changed)

    def test_malformed_realm_spec_fails_inventory(self) -> None:
        with self.assertRaisesRegex(ValueError, "Realm capability names must be string literals"):
            self._render_with_source_change(
                "realm.py", '("realm_read",', '(new_name,')

    def test_messaging_grant_description_matches_host_selection(self) -> None:
        catalog_text = render()
        self.assertIn("| `messaging` | `messaging_send` |", catalog_text)
        self.assertIn("grant the sender messaging;", catalog_text)
        self.assertNotIn("grant the sender messaging_send", catalog_text)
        self.assertEqual([], _select_capabilities(("messaging",), []))
        with self.assertRaisesRegex(ValueError, "Unknown capability grants: messaging_send"):
            _select_capabilities(("messaging_send",), [])
