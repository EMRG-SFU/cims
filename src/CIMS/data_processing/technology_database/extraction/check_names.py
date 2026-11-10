"""Check the names in the extraction layer and the CIMS YAML tree.

Run from the repo root:   python src/technology_database/extraction/check_names.py

The convention (src/technology_database/docs/design.md, Naming):
    '-' joins words within a component.  '_' separates components.

Source data files are author_year_slug[_source]. A sidecar of a source
(<source stem>_meta, _nodes, _flows, _other-values, _measures, ...) carries one
more component naming the sidecar; it is stripped before components are counted
when a file with the source stem sits beside it (the source document in any
format, .docx and .mhtml included, or its main extraction .csv).

Extraction technology names are each source's own labels since the 2026-09-23
archetype restructure: records/source-example.csv maps them to archetypes, and
the database build (src/technology_database/build.py) checks the archetype keys. Here
they only have to follow the shape rule (lowercase hyphenated components joined
by '_'). The controlled equipment/carrier/variant vocabulary in
data/raw_data/technology_database/extraction-vocabulary.csv still applies to the CIMS YAML layer
under general-services.

Folders under data/raw_data/technology_database whose name starts with '_' (_high, _medium, _low and
their _work, _duplicates, triage and scratch contents) are intake working space,
not tracked in git; they are skipped and counted, never silently ignored.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PRIMARY, RAW, RECORDS, ROOT  # noqa: E402

VOCAB_CSV = RAW / "extraction-vocabulary.csv"
STM = ROOT / "cims" / "service-technology-map"   # the CIMS YAML layer; tech_data only, skipped when absent
TEMPLATE = Path(__file__).resolve().parent / "input_template.py"

TOKEN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
DATA_EXT = {".csv", ".xlsx", ".xls", ".xlsm", ".pdf", ".json", ".mhtml"}
# Custom parameters allowed alongside the template's PARAMETERS list. The roster
# fields and the few names the template does not carry are listed here; every
# real parameter is taken from vocabulary.csv, which is what parameter.csv and
# the harmonized layer are built from. Hand-listing them went stale twice --
# abatement-cost and emissions-reduction-potential were live in the database and
# still failed here.
CUSTOM_PARAMS = {
    "heat-output-type", "energy-carrier", "equipment", "variant",
    "cost-basis", "capacity-basis",
}
_PARAMS = RECORDS / "vocabulary" / "parameter.csv"
if _PARAMS.exists():
    import csv as _csv
    with _PARAMS.open(encoding="utf-8-sig", newline="") as _fh:
        CUSTOM_PARAMS |= {r["parameter"] for r in _csv.DictReader(_fh)}
# Directories under service-technology-map that group branches without being
# branch segments themselves (design doc section 3).
GROUPING_DIRS = {"general-services"}

# Only technologies under these top-level service groups are checked against the
# controlled vocabularies; others need only conform to the name shape.
VOCABULARY_SCOPE = {"general-services"}

# Directories whose files keep their publisher's original filenames rather than
# the publisher_year_slug convention (design doc section 8, item 3). Everything
# here is reported as a skip count, never silently ignored.
ORIGINAL_NAME_DIRS = {
    "energy_commodities", "nir", "ceedc", "archive",
    "__marimo__", "__pycache__",
    # superseded/ archives retired artefacts under the names they had when they
    # were retired -- renaming them would break the link to the code and notes
    # that reference them, so the naming rule does not apply inside it.
    "superseded",
}

problems: list[str] = []
checked = {"technology": 0, "parameter": 0, "filename": 0, "folder": 0, "branch": 0}


def fail(kind: str, where: str, msg: str) -> None:
    problems.append(f"[{kind}] {where}: {msg}")


def load_vocabularies() -> dict[str, set[str]]:
    if not VOCAB_CSV.exists():
        sys.exit(f"extraction vocabulary not found at {VOCAB_CSV}")
    vocab: dict[str, set[str]] = {"equipment": set(), "carrier": set(), "variant": set()}
    with VOCAB_CSV.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            vocab.setdefault(r["list"], set()).add(r["term"])
    return vocab


def template_parameters() -> set[str]:
    if not TEMPLATE.exists():
        fail("setup", str(TEMPLATE), "extraction template not found")
        return set()
    src = TEMPLATE.read_text(encoding="utf-8")
    block = src[src.index("PARAMETERS = ["):src.index("]", src.index("PARAMETERS = ["))]
    return set(re.findall(r'\("([^"]+)",', block))


def check_technology(name: str, where: str, vocab: dict[str, set[str]]) -> None:
    checked["technology"] += 1
    parts = name.split("_")
    if not 1 <= len(parts) <= 3:
        fail("technology", where, f"'{name}' has {len(parts)} components, expected 1-3")
        return
    for part in parts:
        if not TOKEN.match(part):
            fail("technology", where, f"'{name}': component '{part}' is not lowercase hyphenated")
            return
    # Slot 1 is always equipment. With three components the rest are carrier
    # then variant. With two, the middle slot is optional (design doc section 5):
    # the second component is whichever of carrier or variant it belongs to,
    # which is unambiguous because the vocabularies are disjoint.
    if parts[0] not in vocab["equipment"]:
        fail("technology", where, f"'{name}': '{parts[0]}' is not in the equipment vocabulary")
    if len(parts) == 3:
        for part, slot in zip(parts[1:], ["carrier", "variant"]):
            if part not in vocab[slot]:
                fail("technology", where, f"'{name}': '{part}' is not in the {slot} vocabulary")
    elif len(parts) == 2:
        p = parts[1]
        if p not in vocab["carrier"] and p not in vocab["variant"]:
            fail("technology", where,
                 f"'{name}': '{p}' is in neither the carrier nor the variant vocabulary")


def check_label(name: str, where: str) -> None:
    """A source's own technology label: shape only (see the module docstring)."""
    checked["technology"] += 1
    for part in name.split("_"):
        if not TOKEN.match(part):
            fail("technology", where, f"'{name}': component '{part}' is not lowercase hyphenated")
            return


def _manifest_stems() -> set[str]:
    """Every document stem the file manifest names (its sidecars are recognised by it,
    with or without the document on the disk)."""
    out = set()
    for name, col in (("source-details_files.csv", "file_on_disk"),
                      ("source-details_reference.csv", "file_on_disk"),
                      ("background-register.csv", "file")):
        p = RAW / name
        if p.exists():
            with p.open(encoding="utf-8-sig", newline="") as fh:
                out |= {Path(r.get(col) or "").name.rsplit(".", 1)[0] for r in csv.DictReader(fh)}
    return out


MANIFEST_STEMS = _manifest_stems()


def twin_dirs(folder: Path) -> list[Path]:
    """A raw-data folder and its primary-sources twin (the document sits in one,
    the extraction in the other)."""
    for a, b in ((RAW, PRIMARY), (PRIMARY, RAW)):
        try:
            return [folder, b / folder.relative_to(a)]
        except ValueError:
            continue
    return [folder]


def in_intake(path: Path) -> bool:
    """Under an intake folder: primary-sources/to-process/, or a working folder whose
    name starts with '_' ('_work', ...)."""
    for base in (RAW, PRIMARY):
        try:
            rel = path.relative_to(base).parts
        except ValueError:
            continue
        return rel[:1] == ("to-process",) or any(p.startswith("_") and not p.startswith("__") for p in rel)
    return False


def main() -> int:
    vocab = load_vocabularies()
    std_params = template_parameters()
    allowed_params = std_params | CUSTOM_PARAMS

    # --- extraction CSVs -------------------------------------------------
    for path in sorted(RAW.rglob("*.csv")):
        # Publisher-original-name directories are exempt from FILENAME checks,
        # not from content checks: an extraction CSV sitting beside its source
        # PDF in one of them still has to use the controlled vocabularies.
        # The header sniff below is what keeps publisher CSVs out of this loop.
        if path.name.endswith("_meta.csv") or in_intake(path):
            continue
        # Cheap header sniff first: the raw-data tree holds multi-hundred-MB
        # source CSVs that are not extractions and must not be read in full.
        try:
            with path.open(encoding="utf-8-sig", newline="") as fh:
                header = fh.readline().strip()
        except (OSError, UnicodeDecodeError):
            continue
        if header.split(",")[:3] != ["process", "technology", "parameter"]:
            continue
        try:
            with path.open(encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.DictReader(fh))
        except (OSError, UnicodeDecodeError):
            continue
        if not rows:
            continue
        rel = path.relative_to(ROOT).as_posix()
        for name in sorted({r["technology"] for r in rows if r["technology"]}):
            check_label(name, rel)
        for name in sorted({r["parameter"] for r in rows if r["parameter"]}):
            checked["parameter"] += 1
            if name not in allowed_params:
                fail("parameter", rel, f"'{name}' is neither a template parameter nor a listed custom one")

    # --- YAML technology files -------------------------------------------
    for path in sorted(STM.rglob("*.yaml")):
        rel = path.relative_to(ROOT).as_posix()
        head = path.read_text(encoding="utf-8-sig").splitlines()[:3]
        fields = {}
        for line in head:
            m = re.match(r'^(technology|branch):\s*"([^"]*)"', line)
            if m:
                fields[m.group(1)] = m.group(2)
        tech, branch = fields.get("technology"), fields.get("branch")
        if tech is None or branch is None:
            fail("yaml", rel, "missing a technology or branch field in the first three lines")
            continue
        if tech != path.stem:
            fail("yaml", rel, f"filename stem '{path.stem}' does not equal technology '{tech}'")
        segs = list(path.relative_to(STM).parts[:-2])   # drop 'technology' and the file
        if segs and segs[0] in VOCABULARY_SCOPE:
            check_technology(tech, rel, vocab)
        else:
            checked["technology"] += 1
            for part in tech.split("_"):
                if not TOKEN.match(part):
                    fail("technology", rel,
                         f"'{tech}': component '{part}' is not lowercase hyphenated")
        # branch == folder path, dotted, dropping grouping dirs and 'technology'
        checked["branch"] += 1
        expected = "." + ".".join(x for x in segs if x not in GROUPING_DIRS)
        if branch != expected:
            fail("branch", rel, f"branch '{branch}' does not match the folder path (expected '{expected}')")

    # --- YAML field and parameter names -----------------------------------
    for path in sorted(STM.rglob("*.yaml")):
        rel = path.relative_to(ROOT).as_posix()
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            m = re.match(r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s*:", line)
            if not m:
                continue
            key = m.group(1)
            checked["parameter"] += 1
            if "_" in key:
                fail("yaml-field", rel,
                     f"'{key}' uses an underscore; field names are single components")

    # --- folder names -----------------------------------------------------
    for base in (RAW, PRIMARY, STM):
        for path in sorted(base.rglob("*")):
            if not path.is_dir() or any(d in path.parts for d in ORIGINAL_NAME_DIRS):
                continue
            if path.name.startswith("__") or in_intake(path):
                continue
            checked["folder"] += 1
            for part in path.name.split("_"):
                if not TOKEN.match(part):
                    fail("folder", path.relative_to(ROOT).as_posix(),
                         f"'{path.name}': component '{part}' is not lowercase hyphenated")

    # --- source data filenames -------------------------------------------
    skipped_original = skipped_intake = 0
    for path in sorted([*RAW.rglob("*"), *PRIMARY.rglob("*")]):
        if not path.is_file() or path.suffix.lower() not in DATA_EXT:
            continue
        if in_intake(path):
            skipped_intake += 1
            continue
        if any(d in path.parts for d in ORIGINAL_NAME_DIRS):
            skipped_original += 1
            continue
        if path.parent in (RAW, PRIMARY):   # repo artefacts, not source documents
            continue
        checked["filename"] += 1
        parts = path.stem.split("_")
        rel = path.relative_to(ROOT).as_posix()
        if parts and parts[-1] == "meta":     # reserved sidecar marker
            parts = parts[:-1]
        elif len(parts) >= 4 and path.suffix.lower() == ".csv" and (
                "_".join(parts[:-1]) in MANIFEST_STEMS or any(
                f != path and f.is_file() for d in twin_dirs(path.parent)
                for f in d.glob("_".join(parts[:-1]) + ".*"))):
            # An extraction sidecar (<source stem>_nodes.csv, _flows.csv, ...): a file
            # with the source stem sits beside it (the document in any format, or the
            # main extraction), so the last component names the sidecar, as '_meta'
            # does, and is not counted.
            parts = parts[:-1]
        if not 3 <= len(parts) <= 4:
            fail("filename", rel,
                 f"expected author_year_slug[_source], got {len(parts)} components")
            continue
        author, year, slug = parts[0], parts[1], parts[2]
        source = parts[3] if len(parts) == 4 else None
        if not re.fullmatch(r"(19|20)\d{2}", year):
            fail("filename", rel, f"'{year}' is not a four-digit year in the second position")
        labelled = [("author", author), ("slug", slug)]
        if source is not None:
            labelled.append(("source", source))
        for label, value in labelled:
            if not TOKEN.match(value):
                fail("filename", rel, f"{label} '{value}' is not lowercase hyphenated")

    # --- report -----------------------------------------------------------
    plural = {"technology": "technology names", "parameter": "parameter names",
              "filename": "filenames", "folder": "folders", "branch": "branch strings"}
    counts = ", ".join(f"{v} {plural[k]}" for k, v in checked.items())
    print(f"checked {counts}")
    print(f"skipped {skipped_original} files in publisher-original-name directories "
          f"({', '.join(sorted(ORIGINAL_NAME_DIRS))})")
    print(f"skipped {skipped_intake} files in intake working folders (primary-sources/to-process, _*)")
    if problems:
        print(f"\n{len(problems)} problem(s):\n")
        for p in problems:
            print("  " + p)
        return 1
    print("\nall names conform to the naming convention (src/technology_database/docs/design.md, section 3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
