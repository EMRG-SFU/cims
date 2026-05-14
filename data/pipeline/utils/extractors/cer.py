"""
Canada Energy Regulator (CER) Extractor Functions

CER data files are named with the edition year in the filename
(e.g. crude-oil-production-2026.csv). find_cer_file() locates the
latest available edition automatically so no code changes are needed
when new data is released.
"""

from pathlib import Path


def find_cer_file(cer_dir: Path, stem: str) -> Path:
    """
    Return the path to the latest CER CSV matching stem-YYYY.csv.

    Files are sorted alphabetically — since the year suffix is zero-padded
    and fixed-width, alphabetical order equals chronological order.

    Parameters
    ----------
    cer_dir : Path
        Directory containing CER CSV files (raw_data/cer).
    stem : str
        Base filename without the year or extension
        (e.g. 'crude-oil-production', 'macro-indicators').

    Raises
    ------
    FileNotFoundError
        If no file matching stem-*.csv exists in cer_dir.
    """
    matches = sorted(cer_dir.glob(f"{stem}-*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No CER file found matching '{stem}-*.csv' in {cer_dir}"
        )
    return matches[-1]
