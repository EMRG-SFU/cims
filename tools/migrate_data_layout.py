"""Move a data folder from the old layout to the five layers (plan Phase 3).

    uv run python tools/migrate_data_layout.py            # show what would move
    uv run python tools/migrate_data_layout.py --apply    # move it

The rename map, tools/data-layout_renames.csv, has one row per file or folder
(old,new,note; paths relative to data/; a trailing / moves a whole folder). An
old path with * matches files by pattern, and {1}, {2}, ... in the new path stand
for what each * matched (the optimizer logs use this). A new path starting with
../ leaves data/: the rest is relative to the repo root (the logs go to
results/logs/, where the code writes them), even when CIMS_DATA_ROOT points
elsewhere. The snapshot tool reads the same map (compare --rename), so a move and
its check use one list. Each commit's map lists only the moves that commit's code
expects.

Run it once in each worktree after pulling. Git has already moved the tracked
files (config/), so the script moves only what git doesn't: raw data, processed
outputs, model inputs, calibration targets and outputs, and policies. It never
overwrites. Where the new path already exists:

- an identical old copy is deleted (for example a file the pull brought into git);
- a different old copy stays where it is, and is listed, so you can compare the two.

It's safe to run again. Uses CIMS_DATA_ROOT if set. At the end it lists anything
left in the old folders that the map doesn't cover, and removes old folders that
are empty (never a file, so tracked .gitkeep files stay until git removes them).
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / 'tools' / 'data-layout_renames.csv'
OLD_TOP = ('raw_data', 'processed_data', 'model_inputs', 'calibration', 'mappings_conversions', 'technology_database')
SKIP = {'.gitkeep', '__pycache__'}


def load_map(path: Path) -> list[tuple[str, str]]:
    with path.open(newline='', encoding='utf-8-sig') as fh:
        return [(r['old'].strip(), r['new'].strip()) for r in csv.DictReader(fh)]


def target(data: Path, new: str) -> Path:
    """A map's new path: under data/, or, for ../, under the repo root."""
    new = new.rstrip('/')
    return ROOT / new[3:] if new.startswith('../') else data / new


def plan_moves(data: Path, pairs: list[tuple[str, str]]):
    """Return (moves, same, differ): (old, new) to move; old copies identical to the
    file already at the new path; and old copies that differ from it."""
    moves, same, differ = [], [], []
    claimed = set()   # sources an earlier row already handles; a folder row leaves them alone

    def one(src: Path, dst: Path):
        if src in claimed:
            return
        claimed.add(src)
        if not dst.exists():
            moves.append((src, dst))
        elif dst.is_file() and src.is_file() and filecmp.cmp(src, dst, shallow=False):
            same.append((src, dst))
        else:
            differ.append((src, dst))

    for old, new in pairs:
        if '*' in old:
            rx = re.compile('^' + '([^/]*)'.join(re.escape(x) for x in old.split('*')) + '$')
            for f in sorted(p for p in data.glob(old) if p.is_file()):
                m = rx.match(f.relative_to(data).as_posix())
                d = target(data, new.format(None, *m.groups()))
                one(f, d / f.name if new.endswith('/') else d)
            continue
        src, dst = data / old.rstrip('/'), target(data, new)
        if not src.exists():
            continue
        if old.endswith('/'):
            # a folder: move it whole if the target is free and no earlier row claimed a
            # file inside it, else file by file
            if not dst.exists() and not any(c == src or src in c.parents for c in claimed):
                moves.append((src, dst))
                claimed.add(src)
                continue
            for f in sorted(p for p in src.rglob('*') if p.is_file() and not SKIP & set(p.parts)):
                one(f, dst / f.relative_to(src))
        else:
            one(src, dst)
    return moves, same, differ


def leftovers(data: Path, keep: set[Path]) -> list[Path]:
    out = []
    for top in OLD_TOP:
        if (data / top).is_dir():
            out += [p for p in (data / top).rglob('*')
                    if p.is_file() and not SKIP & set(p.parts) and p not in keep]
    return sorted(out)


def prune(data: Path) -> None:
    """Remove empty old folders and their __pycache__ folders; never a file."""
    for top in OLD_TOP:
        d = data / top
        if not d.is_dir():
            continue
        for p in sorted((p for p in d.rglob('__pycache__') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            shutil.rmtree(p, ignore_errors=True)
        for p in sorted((p for p in [d, *d.rglob('*')] if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            if not any(p.iterdir()):
                p.rmdir()


def show(p: Path, data: Path) -> str:
    return os.path.relpath(p, data).replace(os.sep, '/')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--apply', action='store_true', help='move the files (default: only show the plan)')
    ap.add_argument('--map', default=str(MAP), help='rename map (default: tools/data-layout_renames.csv)')
    args = ap.parse_args(argv)
    data = Path(os.environ.get('CIMS_DATA_ROOT', ROOT / 'data')).resolve()
    moves, same, differ = plan_moves(data, load_map(Path(args.map)))
    print(f'Data root: {data}\n{len(moves)} to move, {len(same)} already at their new path '
          f'(identical old copy to delete), {len(differ)} at their new path with a different old copy')
    for src, dst in moves[:15]:
        print(f'  {show(src, data)}  ->  {show(dst, data)}')
    if len(moves) > 15:
        print(f'  ... and {len(moves) - 15} more')
    if differ:
        print(f'\nDIFFERENT from the file at the new path, so left where they are ({len(differ)}). '
              f'Compare each pair, keep the right one at the new path, and delete the old one:')
        for src, dst in differ[:40]:
            print(f'  {show(src, data)}  vs  {show(dst, data)}')
    if not args.apply:
        print('\nNothing moved. Run with --apply to move.')
        return 0
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    for src, _ in same:
        src.unlink()
    prune(data)
    left = leftovers(data, keep={src for src, _ in differ})
    if left:
        print(f'\nLeft in the old folders, not in the map ({len(left)}). Move or delete these by hand:')
        for p in left[:40]:
            print(f'  {show(p, data)}')
    print('\nDone.' if not (left or differ) else '\nDone, with the items above to check.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
