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
    "scikit-learn",
    "joblib",
    "threadpoolctl",
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
        "scipy_blas": scipy_blas(),
    }


def scipy_blas() -> dict[str, str]:
    """Record the numerical backend that affects seeded paper computations."""
    from scipy.__config__ import CONFIG
    backend = CONFIG.get("Build Dependencies", {}).get("blas", {})
    return {key: str(backend.get(key, "unknown")) for key in ("name", "version")}


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


def publication_style() -> None:
    """Set the common figure typography without requiring system fonts."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "text.usetex": False, "font.family": "serif",
        "font.serif": ["Palatino", "TeX Gyre Pagella", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 8, "axes.labelsize": 9,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "axes.spines.top": True,
        "axes.spines.right": True, "axes.linewidth": 1.,
        "xtick.major.width": .9, "ytick.major.width": .9,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "savefig.bbox": "tight", "savefig.pad_inches": .02,
    })


def equal_legend_rows(legend, spacing_points: float = 15.1020833333333) -> None:
    """Keep method names and markers aligned when saving at another DPI."""
    from matplotlib.mathtext import MathTextParser
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory

    figure = legend.get_figure()
    title = legend.get_title()
    title.set_transform(title.get_transform() + ScaledTranslation(
        0, .1140625 / 72, figure.dpi_scale_trans))
    parser = MathTextParser("path")
    for index, (text, handle) in enumerate(zip(legend.get_texts(), legend.legend_handles)):
        offset = (parser.parse(text.get_text(), dpi=72, prop=text.get_fontproperties()).glyphs[0][4]
                  if "$" in text.get_text() else 0)
        baseline = -13 - index * spacing_points
        text.set_transform(blended_transform_factory(text.get_transform(), title.get_transform())
                           + ScaledTranslation(0, (baseline-offset)/72, figure.dpi_scale_trans))
        handle.set_ydata([0.0] * len(handle.get_ydata()))
        handle.set_transform(blended_transform_factory(handle.get_transform(), title.get_transform())
                             + ScaledTranslation(0, (baseline+2.8)/72, figure.dpi_scale_trans))
