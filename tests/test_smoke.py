from __future__ import annotations

import importlib
import pkgutil

import pytest


def test_every_module_imports():
    """Import walk: a broken import anywhere fails the build, not the demo."""
    import factory

    failures = []
    for mod in pkgutil.walk_packages(factory.__path__, prefix="factory."):
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{mod.name}: {exc}")
    assert not failures, "\n".join(failures)


def test_cli_is_importable():
    from factory.cli import main

    assert callable(main)
