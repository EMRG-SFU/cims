"""Snapshot the pipeline's outputs, and compare snapshots (plan §6).

A snapshot records every file under data/processed, data/model, and
data/config/model-default: its row count, its columns, and a hash of its rows sorted. It also
keeps a compressed copy of each file, so a comparison can show what changed.

    uv run python tools/output_snapshot.py record baseline --note "Phase 0 baseline"
    uv run python tools/output_snapshot.py compare baseline              # with the current outputs
    uv run python tools/output_snapshot.py compare baseline other        # two snapshots
    uv run python tools/output_snapshot.py compare baseline --rename tools/renames.csv
    uv run python tools/output_snapshot.py list

Snapshots live beside the worktrees, in <worktree>/../snapshots/<name>/ (set
CIMS_SNAPSHOT_ROOT to put them elsewhere), so every worktree compares against the same
baseline. The data folder follows the code, as in controls_conversions: <worktree>/data,
unless CIMS_DATA_ROOT is set.

Each snapshot has a content id: snapshots of identical outputs share it. A commit that
only restructures says "outputs identical: snapshot <id>".

How files are compared:
  - identical: the same rows, in the same order;
  - reordered: the same rows, in another order;
  - within 1e-9: numbers differ only at 1e-9 relative or less (for example 0.1 against
    0.10000000000000001);
  - changed: anything else. Rows are matched on their text cells (and Year), and the
    report shows which numbers moved and by how much.

A rename map is a CSV with the columns old,new, with paths relative to data/. A path
ending in / renames a folder. Files are matched under their new names.

compare exits with 1 when any file changed, was added, or was removed, and 0 otherwise.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CIMS_DATA_ROOT", ROOT / "data"))
SNAPSHOTS = Path(os.environ.get("CIMS_SNAPSHOT_ROOT", ROOT.parent / "snapshots"))
FOLDERS = ("processed", "model", "config/model-default")
SKIP = {".gitkeep"}
KEY_COLUMNS = {"year"}  # numeric, but they identify a row rather than measure something
REL = 1e-9
SEP = "\x1f"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


# ---- reading files ----------------------------------------------------------------------

def _text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def parse(name: str, raw: bytes) -> tuple[list[str], list[list[str]]]:
    """A file's header and rows. A workbook gives one row per sheet row, starting with
    the sheet's name; any other file that isn't a CSV gives one row per line."""
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        reader = csv.reader(io.StringIO(_text(raw), newline=""))
        header = next(reader, [])
        return header, list(reader)
    if suffix in (".xlsx", ".xlsm"):
        import openpyxl

        book = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
        rows = [[sheet.title, *map(_cell, row)] for sheet in book.worksheets
                for row in sheet.iter_rows(values_only=True)]
        book.close()
        width = max((len(r) for r in rows), default=1)
        return ["sheet", *(f"column {i}" for i in range(1, width))], [r + [""] * (width - len(r)) for r in rows]
    return ["line"], [[line] for line in _text(raw).splitlines()]


def fingerprint(header: list[str], rows: list[list[str]]) -> dict:
    lines = [SEP.join(r).encode("utf-8", "surrogatepass") for r in rows]
    head = SEP.join(header).encode("utf-8", "surrogatepass")
    ordered, unordered = hashlib.sha256(head), hashlib.sha256(head)
    for line in lines:
        ordered.update(b"\n" + line)
    for line in sorted(lines):
        unordered.update(b"\n" + line)
    return {"rows": len(rows), "columns": header,
            "sorted": unordered.hexdigest(), "ordered": ordered.hexdigest()}


def output_files(data: Path):
    for folder in FOLDERS:
        base = data / folder
        if not base.is_dir():
            print(f"warning: {base} does not exist", file=sys.stderr)
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.name not in SKIP:
                yield path.relative_to(data).as_posix(), path


def content_id(files: dict) -> str:
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{rel}\n{files[rel]['sorted']}\n".encode())
    return h.hexdigest()[:10]


def git_state() -> dict:
    def run(*args):
        try:
            done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60)
            return done.stdout.strip() if done.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted": run("status", "--porcelain", "--untracked-files=no").splitlines()}


# ---- snapshots ----------------------------------------------------------------------------

class Snapshot:
    def __init__(self, name: str):
        self.dir = SNAPSHOTS / name
        manifest = self.dir / "manifest.json"
        if not manifest.is_file():
            sys.exit(f"no snapshot named {name!r} in {SNAPSHOTS} (see: output_snapshot.py list)")
        self.meta = json.loads(manifest.read_text(encoding="utf-8"))
        self.files = self.meta["files"]
        git = self.meta.get("git", {})
        self.label = (f"snapshot {name} (id {self.meta['id']}; {git.get('branch') or '?'} at "
                      f"{(git.get('commit') or '?')[:8]}; recorded {self.meta['recorded']})")

    def load(self, rel: str):
        return parse(rel, gzip.decompress((self.dir / "files" / f"{rel}.gz").read_bytes()))


class Current:
    def __init__(self):
        self.files = {rel: fingerprint(*parse(rel, path.read_bytes())) for rel, path in output_files(DATA)}
        self.label = f"the current outputs in {DATA} (id {content_id(self.files)})"

    def load(self, rel: str):
        return parse(rel, (DATA / rel).read_bytes())


def record(name: str, note: str, force: bool) -> None:
    dest, partial = SNAPSHOTS / name, SNAPSHOTS / f"{name}.partial"
    if dest.exists():
        if not force:
            sys.exit(f"snapshot {name!r} already exists in {SNAPSHOTS}; pass --force to replace it")
        shutil.rmtree(dest)
    if partial.exists():
        shutil.rmtree(partial)
    files, totals = {}, defaultdict(lambda: [0, 0, 0])
    for rel, path in output_files(DATA):
        raw = path.read_bytes()
        files[rel] = fingerprint(*parse(rel, raw)) | {"bytes": len(raw)}
        out = partial / "files" / f"{rel}.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out, "wb", compresslevel=6) as fh:
            fh.write(raw)
        t = totals[rel.split("/")[0]]
        t[0] += 1; t[1] += files[rel]["rows"]; t[2] += len(raw)
    if not files:
        sys.exit(f"no output files under {DATA}")
    meta = {"name": name, "id": content_id(files), "recorded": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "note": note, "worktree": str(ROOT), "data": str(DATA), "git": git_state(), "files": files}
    partial.mkdir(parents=True, exist_ok=True)
    (partial / "manifest.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    partial.rename(dest)
    print(f"Recorded snapshot {name} (id {meta['id']}) in {dest}")
    for folder, (n, rows, size) in totals.items():
        print(f"  {folder:<16}{n:>5} files {rows:>11,} rows {size / 1e6:>8.1f} MB")
    if meta["git"]["uncommitted"]:
        print(f"  note: {len(meta['git']['uncommitted'])} tracked files had uncommitted changes")


def list_snapshots() -> None:
    if not SNAPSHOTS.is_dir():
        print(f"no snapshots yet ({SNAPSHOTS})")
        return
    for manifest in sorted(SNAPSHOTS.glob("*/manifest.json")):
        m = json.loads(manifest.read_text(encoding="utf-8"))
        git = m.get("git", {})
        print(f"{m['name']:<24} id {m['id']}  {m['recorded']}  {git.get('branch') or '?'} at "
              f"{(git.get('commit') or '?')[:8]}  {len(m['files'])} files  {m.get('note') or ''}")


# ---- comparing ----------------------------------------------------------------------------

def _num(value: str):
    try:
        return float(value)
    except ValueError:
        return None


def _close(a: float, b: float) -> bool:
    if a == b or (math.isnan(a) and math.isnan(b)):
        return True
    return abs(a - b) <= REL * max(abs(a), abs(b))


def diff_table(old, new, examples: int) -> dict:
    (h1, r1), (h2, r2) = old, new
    result = {"rows": (len(r1), len(r2))}
    if h1 != h2:
        result["columns"] = ([c for c in h1 if c not in h2], [c for c in h2 if c not in h1])
    common = [c for c in h1 if c in h2]
    i1, i2 = [h1.index(c) for c in common], [h2.index(c) for c in common]
    p1 = [tuple(r[i] if i < len(r) else "" for i in i1) for r in r1]
    p2 = [tuple(r[i] if i < len(r) else "" for i in i2) for r in r2]

    # Rows in both files, as text, need no further look.
    left, right = Counter(p1) - Counter(p2), Counter(p2) - Counter(p1)

    def unmatched(rows, counts):
        out = []
        for row in rows:
            if counts[row] > 0:
                counts[row] -= 1
                out.append(row)
        return out
    u1, u2 = unmatched(p1, left), unmatched(p2, right)

    numeric = [j for j, c in enumerate(common) if c.strip().lower() not in KEY_COLUMNS
               and any(r[j] != "" for r in u1 + u2)
               and all(v == "" or _num(v) is not None for r in u1 + u2 for v in (r[j],))]
    key_cols = [j for j in range(len(common)) if j not in numeric]
    groups1, groups2 = defaultdict(list), defaultdict(list)
    for r in u1:
        groups1[tuple(r[j] for j in key_cols)].append(r)
    for r in u2:
        groups2[tuple(r[j] for j in key_cols)].append(r)

    only_old = only_new = values = rows_changed = 0
    largest, by_column, shown = 0.0, Counter(), []
    for key in groups1.keys() | groups2.keys():
        a, b = groups1.get(key, []), groups2.get(key, [])
        only_old += max(0, len(a) - len(b))
        only_new += max(0, len(b) - len(a))
        for x, y in zip(a, b):
            moved = False
            for j in numeric:
                if x[j] == y[j]:
                    continue
                fx, fy = _num(x[j]), _num(y[j])
                if fx is not None and fy is not None and _close(fx, fy):
                    continue
                rel = (abs(fx - fy) / max(abs(fx), abs(fy))) if fx is not None and fy is not None else math.inf
                values += 1
                moved = True
                by_column[common[j]] += 1
                largest = max(largest, rel)
                if len(shown) < examples:
                    label = ", ".join(f"{common[k]}={x[k]}" for k in key_cols if x[k] != "")
                    shown.append(f"[{label[:240]}] {common[j]}: {x[j] or '(blank)'} -> {y[j] or '(blank)'}")
            rows_changed += moved
    if only_old or only_new:
        for rows, sign in ((u1, "-"), (u2, "+")):
            other = groups2 if sign == "-" else groups1
            for r in rows:
                if len(shown) >= examples:
                    break
                if tuple(r[j] for j in key_cols) not in other:
                    shown.append(f"{sign} {', '.join(f'{common[k]}={r[k]}' for k in range(len(common)) if r[k] != '')[:240]}")
    result |= {"only_old": only_old, "only_new": only_new, "values": values, "rows_changed": rows_changed,
               "largest": largest, "by_column": by_column, "examples": shown}
    result["within"] = not (h1 != h2 or only_old or only_new or values)
    return result


def describe(rel: str, d: dict) -> list[str]:
    parts = []
    if d["rows"][0] != d["rows"][1]:
        parts.append(f"rows {d['rows'][0]:,} -> {d['rows'][1]:,}")
    if "columns" in d:
        gone, added = d["columns"]
        parts.append("columns" + (f" removed {gone}" if gone else "") + (f" added {added}" if added else "")
                     + ("" if gone or added else " reordered"))
    if d["only_old"] or d["only_new"]:
        parts.append(f"{d['only_old']:,} rows only before, {d['only_new']:,} only after")
    if d["values"]:
        cols = ", ".join(f"{c} {n:,}" for c, n in d["by_column"].most_common(4))
        size = "a blank or text" if math.isinf(d["largest"]) else f"{d['largest']:.2g}"
        parts.append(f"{d['values']:,} values in {d['rows_changed']:,} rows ({cols}); largest change {size} relative")
    return [f"  {rel}: " + "; ".join(parts)] + [f"      {e}" for e in d["examples"]]


def compare(old_name: str, new_name: str | None, rename: str | None, examples: int, verbose: bool) -> int:
    old = Snapshot(old_name)
    new = Snapshot(new_name) if new_name else Current()
    pairs = []
    if rename:
        with open(rename, newline="", encoding="utf-8-sig") as fh:
            pairs = [(r["old"].strip(), r["new"].strip()) for r in csv.DictReader(fh)]

    def renamed(rel):
        for a, b in pairs:
            if (a.endswith("/") and rel.startswith(a)) or rel == a:
                return b + rel[len(a):]
        return rel
    old_by_new = {renamed(rel): rel for rel in old.files}
    names = sorted(old_by_new.keys() | new.files.keys())

    status, details = {}, {}
    for rel in names:
        if rel not in new.files:
            status[rel] = "removed"
        elif rel not in old_by_new:
            status[rel] = "added"
        else:
            a, b = old.files[old_by_new[rel]], new.files[rel]
            if a["ordered"] == b["ordered"]:
                status[rel] = "identical"
            elif a["sorted"] == b["sorted"]:
                status[rel] = "reordered"
            else:
                details[rel] = diff_table(old.load(old_by_new[rel]), new.load(rel), examples)
                status[rel] = "within 1e-9" if details[rel]["within"] else "changed"

    kinds = ["identical", "reordered", "within 1e-9", "changed", "removed", "added"]
    print(f"Before: {old.label}\nAfter:  {new.label}" + (f"\nRenames: {rename}" if rename else "") + "\n")
    print(f"{'':<16}{'files':>7}" + "".join(f"{k:>13}" for k in kinds))
    table = defaultdict(Counter)
    for rel, s in status.items():
        table[rel.split("/")[0]][s] += 1
    for folder in [*FOLDERS, *sorted(set(table) - set(FOLDERS))]:
        if folder in table:
            print(f"{folder:<16}{sum(table[folder].values()):>7}" + "".join(f"{table[folder][k]:>13}" for k in kinds))
    total = Counter(status.values())
    print(f"{'total':<16}{len(status):>7}" + "".join(f"{total[k]:>13}" for k in kinds))

    for kind in ("changed", "removed", "added", "reordered", "within 1e-9"):
        rels = [r for r, s in status.items() if s == kind]
        if not rels:
            continue
        if kind in ("reordered", "within 1e-9") and not verbose:
            print(f"\n{kind.capitalize()}: {len(rels)} files (--verbose lists them)")
            continue
        print(f"\n{kind.capitalize()} ({len(rels)}):")
        for rel in rels:
            if kind == "changed":
                print("\n".join(describe(rel, details[rel])))
            else:
                print(f"  {rel}" + (f"  (was {old_by_new[rel]})" if rel in old_by_new and old_by_new[rel] != rel else ""))
    same = not (total["changed"] or total["removed"] or total["added"])
    print("\nOutputs identical" + (" (some files reordered or within 1e-9)" if total["reordered"] or total["within 1e-9"] else "")
          + f": snapshot {old.meta['id']}." if same else f"\nOutputs differ from snapshot {old.meta['id']}.")
    return 0 if same else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="snapshot the current outputs")
    rec.add_argument("name")
    rec.add_argument("--note", default="", help="a line saying what this snapshot is")
    rec.add_argument("--force", action="store_true", help="replace a snapshot with the same name")
    cmp_ = sub.add_parser("compare", help="compare a snapshot with the current outputs or another snapshot")
    cmp_.add_argument("before")
    cmp_.add_argument("after", nargs="?", help="a second snapshot (default: the current outputs)")
    cmp_.add_argument("--rename", help="CSV of old,new paths relative to data/")
    cmp_.add_argument("--examples", type=int, default=3, help="example changes to show per file (default 3)")
    cmp_.add_argument("--verbose", action="store_true", help="also list reordered and within-1e-9 files")
    sub.add_parser("list", help="list the snapshots")
    args = parser.parse_args()
    if args.command == "record":
        record(args.name, args.note, args.force)
    elif args.command == "compare":
        return compare(args.before, args.after, args.rename, args.examples, args.verbose)
    else:
        list_snapshots()
    return 0


if __name__ == "__main__":
    sys.exit(main())
