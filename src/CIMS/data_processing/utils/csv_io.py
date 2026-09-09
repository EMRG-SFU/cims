"""
BOM-tolerant CSV reading.

Some hand-curated inputs under ``raw_data/fixed_data/`` carry more than one
UTF-8 BOM -- a file re-saved by a tool that read it as plain UTF-8 (keeping the
BOM as ordinary text) and then wrote it back with ``utf-8-sig`` gains a second
one. Python's ``utf-8-sig`` codec strips exactly one, so the leftover shows up
glued to the first column name: ``csv.DictReader`` yields ``'﻿Branch'``
instead of ``'Branch'`` and every ``row['Branch']`` raises ``KeyError``.

The helpers here strip *every* leading BOM, so one stray copy in an input file
cannot take a pipeline script down.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

BOM = '﻿'

# Order matters: try strict UTF-8 first, fall back for legacy exports.
_ENCODINGS = ('utf-8-sig', 'utf-8', 'latin-1')


def strip_bom(text: str) -> str:
    """Remove every leading BOM from ``text`` (not just the first)."""
    while text.startswith(BOM):
        text = text[len(BOM):]
    return text


def clean_fieldnames(fieldnames):
    """Strip stray BOMs and surrounding whitespace from CSV header names."""
    if fieldnames is None:
        return None
    return [strip_bom(name).strip() if isinstance(name, str) else name
            for name in fieldnames]


def read_text(path) -> str:
    """Read a CSV as text with encoding fallbacks, with all leading BOMs gone."""
    path = Path(path)
    for encoding in _ENCODINGS:
        try:
            return strip_bom(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return strip_bom(path.read_text(encoding='latin-1', errors='replace'))


def read_dict_rows(path) -> list:
    """
    Parse a CSV into a list of dicts, with BOM-free field names.

    Drop-in replacement for::

        with open(path, encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
    """
    return list(iter_dict_rows(path))


def iter_dict_rows(path) -> Iterator[dict]:
    """Streaming form of :func:`read_dict_rows`."""
    text = read_text(path)
    reader = csv.DictReader(text.splitlines())
    reader.fieldnames = clean_fieldnames(reader.fieldnames)
    yield from reader
