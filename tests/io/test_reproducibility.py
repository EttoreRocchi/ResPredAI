"""Tests for reproducibility manifest."""

import json
from pathlib import Path

from respredai.io.reproducibility import (
    get_package_versions,
    hash_file,
    save_reproducibility_manifest,
)


class TestHashFile:
    """Tests for hash_file function."""

    def test_hash_file_returns_sha256(self, tmp_path):
        """Test that hash_file returns a valid SHA256 hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        result = hash_file(test_file)
        assert len(result) == 64  # SHA256 hex length
        assert result.isalnum()

    def test_hash_file_deterministic(self, tmp_path):
        """Test that hash_file produces the same hash for the same content."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        assert hash_file(test_file) == hash_file(test_file)

    def test_hash_file_different_content(self, tmp_path):
        """Test that different content produces different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("hello world")
        file2.write_text("different content")
        assert hash_file(file1) != hash_file(file2)


class TestGetPackageVersions:
    """Tests for get_package_versions function."""

    def test_returns_dict(self):
        """Test that get_package_versions returns a dictionary."""
        versions = get_package_versions()
        assert isinstance(versions, dict)

    def test_contains_numpy(self):
        """Test that numpy version is included."""
        versions = get_package_versions()
        assert "numpy" in versions

    def test_contains_pandas(self):
        """Test that pandas version is included."""
        versions = get_package_versions()
        assert "pandas" in versions

    def test_contains_sklearn(self):
        """Test that scikit-learn version is included."""
        versions = get_package_versions()
        assert "scikit-learn" in versions


class TestSaveReproducibilityManifest:
    """Tests for save_reproducibility_manifest function."""

    def test_saves_json(self, tmp_path):
        """Test that manifest is saved as valid JSON."""
        manifest = {"test": "data", "nested": {"key": "value"}}
        path = save_reproducibility_manifest(manifest, tmp_path)
        assert path.exists()
        assert path.name == "reproducibility.json"
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == manifest

    def test_creates_file_in_correct_location(self, tmp_path):
        """Test that file is created in the specified directory."""
        manifest = {"test": "data"}
        path = save_reproducibility_manifest(manifest, tmp_path)
        assert path.parent == tmp_path

    def test_handles_non_serializable_types(self, tmp_path):
        """Test that non-JSON-serializable types are handled via default=str."""
        manifest = {"path": Path("/some/path"), "test": "data"}
        path = save_reproducibility_manifest(manifest, tmp_path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["path"] == "/some/path"
