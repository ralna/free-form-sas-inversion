"""
Shared test scaffolding.

The mock-based tests never run the real solver -- they monkeypatch
`ffsi.api.optimize` -- so they only need the *import* to succeed. When galahad is
missing we install a stub module so the API is importable, and skip the tests
that genuinely require a real solve.
"""
import sys
import types

import pytest

try:
    import galahad  # noqa: F401
    GALAHAD_AVAILABLE = True
except ImportError:
    GALAHAD_AVAILABLE = False
    # Its attributes are never invoked: every test that would reach the solver
    # either mocks `ffsi.api.optimize` or is skipped below.
    _stub = types.ModuleType("galahad")
    _stub.snls = types.SimpleNamespace()
    sys.modules["galahad"] = _stub


def pytest_collection_modifyitems(config, items):
    """Skip the real-solve tests when galahad is only a stub."""
    if GALAHAD_AVAILABLE:
        return
    skip_real_solve = pytest.mark.skip(reason="requires galahad (real GALAHAD solve)")
    for item in items:
        # If GALAHAD isn't available
        # Only test class:TestModelResolution as it doesn't require GALAHAD
        if "test_api.py" in item.nodeid and "TestModelResolution" not in item.nodeid:
            item.add_marker(skip_real_solve)
