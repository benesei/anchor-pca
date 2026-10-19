"""Load the Fama--French 49 Industry Portfolios for the paper's periods."""

from __future__ import annotations

import hashlib
import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATA_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/49_Industry_Portfolios_daily_CSV.zip"
)
ARCHIVE_NAME = "49_Industry_Portfolios_daily_CSV.zip"
ARCHIVE_SHA256 = "e214c54a41f058c03ed4a4e582b2126e04120d2a84bc19c023bd2a4251f77097"
CSV_MEMBER = "49_Industry_Portfolios_Daily.csv"
N_FEATURES = 49
MISSING_SENTINELS = (-99.99, -999.0)
SOURCE_PERIODS = ((1970, 1974), (1975, 1979), (1980, 1984), (1985, 1989), (1990, 1994))
TARGET_PERIODS = (
    (1995, 1999),
    (2000, 2004),
    (2005, 2009),
    (2010, 2014),
    (2015, 2019),
    (2020, 2024),
)


@dataclass(frozen=True)
class Period:
    period_id: str
    role: str
    start_year: int
    end_year: int
    values: np.ndarray
    raw_rows: int
    incomplete_rows_removed: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_archive(archive: Path | None, data_dir: Path) -> Path:
    """Use a supplied archive or download the current official data."""
    if archive is None:
        data_dir.mkdir(parents=True, exist_ok=True)
        archive = data_dir / ARCHIVE_NAME
        if not archive.exists():
            print(f"Downloading {DATA_URL}")
            urllib.request.urlretrieve(DATA_URL, archive)
    archive = Path(archive)
    if not archive.is_file():
        raise FileNotFoundError(f"Fama--French archive not found: {archive}")
    return archive


def archive_provenance(archive: Path) -> dict[str, object]:
    """Identify the downloaded revision without requiring the original vintage."""
    with zipfile.ZipFile(archive) as zipped:
        matches = [name for name in zipped.namelist() if Path(name).name == CSV_MEMBER]
        if len(matches) != 1:
            raise RuntimeError(f"Archive must contain exactly one {CSV_MEMBER} file.")
        contents = zipped.read(matches[0])
    revision = re.search(r"(\d{6}) CRSP", contents.decode("utf-8-sig"))
    return {
        "archive_sha256": sha256_file(archive),
        "csv_sha256": hashlib.sha256(contents).hexdigest(),
        "crsp_revision": revision.group(1) if revision else None,
        "original_archive": sha256_file(archive) == ARCHIVE_SHA256,
    }


def load_periods(archive: Path) -> tuple[Period, ...]:
    """Read the value-weighted daily-return table and form five-year periods."""
    with zipfile.ZipFile(archive) as zipped:
        matches = [name for name in zipped.namelist() if Path(name).name == CSV_MEMBER]
        if len(matches) != 1:
            raise RuntimeError(f"Archive must contain exactly one {CSV_MEMBER} file.")
        csv_bytes = zipped.read(matches[0])
    lines = csv_bytes.decode("utf-8-sig").splitlines()
    labels = [
        index
        for index, line in enumerate(lines)
        if line.strip() == "Average Value Weighted Returns -- Daily"
    ]
    if len(labels) != 1:
        raise RuntimeError("Could not identify the value-weighted daily-return table.")
    header = labels[0] + 1
    end = next(
        (index for index in range(header + 1, len(lines)) if not lines[index].strip()),
        len(lines),
    )
    frame = pd.read_csv(io.StringIO("\n".join(lines[header:end])), skipinitialspace=True)
    if frame.shape[1] != N_FEATURES + 1:
        raise RuntimeError("The value-weighted table must contain 49 return series.")
    dates = pd.to_datetime(frame.iloc[:, 0].astype(str), format="%Y%m%d").to_numpy(
        dtype="datetime64[D]"
    )
    if len(np.unique(dates)) != len(dates) or np.any(dates[1:] <= dates[:-1]):
        raise RuntimeError("Daily observations must have unique increasing dates.")
    values = frame.iloc[:, 1:].to_numpy(dtype=float)
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    incomplete = np.any(
        np.logical_or.reduce(
            tuple(np.isclose(values, sentinel) for sentinel in MISSING_SENTINELS)
        ),
        axis=1,
    )

    periods = []
    all_periods = (("source", SOURCE_PERIODS), ("target", TARGET_PERIODS))
    for role, definitions in all_periods:
        for index, (start_year, end_year) in enumerate(definitions, start=1):
            raw_mask = (years >= start_year) & (years <= end_year)
            keep = raw_mask & ~incomplete
            selected = np.asarray(values[keep], dtype=float)
            if selected.shape[0] < 2 or not np.all(np.isfinite(selected)):
                raise RuntimeError(f"Invalid observations in {start_year}--{end_year}.")
            periods.append(
                Period(
                    period_id=f"{role}_{index}",
                    role=role,
                    start_year=start_year,
                    end_year=end_year,
                    values=selected,
                    raw_rows=int(raw_mask.sum()),
                    incomplete_rows_removed=int(raw_mask.sum() - keep.sum()),
                )
            )
    return tuple(periods)


def period_data_hash(periods: tuple[Period, ...]) -> str:
    """Hash only the observations used, ignoring later additions to the archive."""
    digest = hashlib.sha256()
    for period in periods:
        digest.update(f"{period.start_year}-{period.end_year}".encode())
        digest.update(np.asarray(period.values, dtype="<f8").tobytes())
    return digest.hexdigest()
