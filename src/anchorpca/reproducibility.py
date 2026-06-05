"""Small helpers for experiment metadata."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import platform
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_SOFTWARE_PACKAGES = (
    "anchorpca",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "pytest",
    "torch",
    "minPCA",
)


def software_versions(packages: Iterable[str] = DEFAULT_SOFTWARE_PACKAGES) -> dict[str, object]:
    """Return Python, platform, and package versions for reproducibility metadata.

    Missing optional packages are recorded as ``None`` rather than imported.
    This avoids side effects from importing plotting or optimizer libraries.
    """

    package_versions: dict[str, str | None] = {}
    for package in packages:
        try:
            package_versions[str(package)] = importlib_metadata.version(str(package))
        except importlib_metadata.PackageNotFoundError:
            package_versions[str(package)] = None

    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": package_versions,
    }


def find_repo_root(start: str | Path | None = None) -> Path | None:
    """Find the repository root by looking for the local package layout."""

    current = Path.cwd() if start is None else Path(start)
    current = current.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "anchorpca").exists():
            return candidate
    return None


def repo_relative_path(path: str | Path, *, repo_root: str | Path | None = None) -> str:
    """Return a stable repo-relative path for metadata when possible."""

    resolved = Path(path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root(resolved)
    if root is not None:
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            pass
    return str(path)
