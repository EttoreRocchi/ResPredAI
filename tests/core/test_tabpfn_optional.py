"""Tests for the TabPFN optional-dependency and token-validation path."""

import subprocess
import sys
import textwrap

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


class TestTorchIsOptional:
    """torch arrives with the tabpfn extra only; the base package must not need it."""

    def test_package_imports_without_torch(self):
        code = textwrap.dedent(
            """
            import importlib.abc
            import sys

            class BlockTorch(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name == "torch" or name.startswith("torch."):
                        raise ModuleNotFoundError(f"No module named {name!r}")
                    return None

            sys.meta_path.insert(0, BlockTorch())
            import respredai, respredai.cli, respredai.core.model_builder
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
