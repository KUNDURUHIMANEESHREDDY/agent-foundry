"""Test package marker.

This file exists solely so `tests` is an importable package. `conftest.py` and
`test_langfuse.py` do `from tests.langfuse_helpers import ...`, and without this
marker pytest's default `prepend` import mode puts `tests/` itself on
`sys.path` rather than the repository root -- so `import tests...` fails.

It failed in CI on Ubuntu and passed locally, which is the interesting part:
`python -m pytest` prepends the current directory to `sys.path` and masks the
problem, while the bare `pytest tests -q` that CI actually runs does not. Delete
this file and the suite still passes locally while breaking on a clean checkout.

With the marker present, pytest walks up past `tests/` to the repository root,
puts *that* on `sys.path`, and imports the modules as `tests.test_langfuse`.
"""