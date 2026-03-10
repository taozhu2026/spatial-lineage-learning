from __future__ import annotations

import unittest

from spatial_lineage.stable.core.registry import Registry


class RegistryTestCase(unittest.TestCase):
    def test_registry_register_and_build(self) -> None:
        registry: Registry[object] = Registry("dummy")

        @registry.register("example")
        class Example:
            def __init__(self, value: int) -> None:
                self.value = value

        instance = registry.build("example", value=7)
        self.assertEqual(instance.value, 7)

    def test_registry_unknown_key_raises(self) -> None:
        registry: Registry[object] = Registry("dummy")
        with self.assertRaises(KeyError):
            registry.get("missing")
