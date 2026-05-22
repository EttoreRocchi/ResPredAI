"""Tests for the TabPFN optional-dependency and token-validation path."""

import pytest

from respredai.core.model_builder import TABPFN_TOKEN_ENV, ensure_tabpfn_available


class TestEnsureTabpfnAvailable:
    """Validate the eager TabPFN setup checks."""

    def test_raises_runtime_error_without_token(self, monkeypatch):
        pytest.importorskip("tabpfn")
        monkeypatch.delenv(TABPFN_TOKEN_ENV, raising=False)
        with pytest.raises(RuntimeError, match=TABPFN_TOKEN_ENV):
            ensure_tabpfn_available()

    def test_passes_with_token(self, monkeypatch):
        pytest.importorskip("tabpfn")
        monkeypatch.setenv(TABPFN_TOKEN_ENV, "dummy-not-validated-here")
        ensure_tabpfn_available()
