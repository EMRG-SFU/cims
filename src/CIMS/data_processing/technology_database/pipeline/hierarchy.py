"""The CIMS hierarchy: the definitive list of services and end uses.

records/cims-hierarchy.csv is a snapshot of the hierarchy as the CIMS project
keeps it. Replace the file with the latest copy whenever it changes; the build
then

  * stamps the snapshot it used (row count and a content hash) in
    output/hierarchy-stamp.csv,
  * reports the nodes added and removed since the last build, against the copy
    output/cims-hierarchy.csv, the copy the last build wrote (views.py rewrites it
    after this comparison), and
  * fails if any archetype-service row or annotation points at a path the
    snapshot no longer has.

A path in the records is either a node exactly as the hierarchy writes it
(`<sector>.iron-and-steel.service.hot-metal`) or a pattern in which `*` stands
for exactly one segment (`<sector>.*.service.steam`, every sector's steam
node). A pattern must match at least one node.
"""
from __future__ import annotations

import hashlib

from .common import HIERARCHY, OUTPUT, read, write

# written by views.build(); read here before this build overwrites it
PREVIOUS = OUTPUT / "cims-hierarchy.csv"


class Hierarchy:
    def __init__(self, path=HIERARCHY):
        self.path = path
        self.nodes = [r["branch"].strip() for r in read(path) if r.get("branch", "").strip()]
        self.set = set(self.nodes)
        self.leaves = [b for b in self.nodes
                       if not any(o.startswith(b + ".") for o in self.nodes)]
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else ""

    def expand(self, pattern: str) -> list[str]:
        """The nodes a path or pattern names; [] when it names none."""
        pattern = pattern.strip()
        if "*" not in pattern:
            return [pattern] if pattern in self.set else []
        want = pattern.split(".")
        return [b for b in self.nodes
                if len(b.split(".")) == len(want)
                and all(w == "*" or w == s for w, s in zip(want, b.split(".")))]

    @staticmethod
    def short(path: str) -> str:
        """A path without the `<sector>.` placeholder, for display."""
        return path[len("<sector>."):] if path.startswith("<sector>.") else path


def stamp(h: Hierarchy) -> dict:
    """Record the snapshot used and what changed since the last build."""
    prev = [r["branch"].strip() for r in read(PREVIOUS)]
    added = sorted(h.set - set(prev)) if prev else []
    removed = sorted(set(prev) - h.set) if prev else []
    write(OUTPUT / "hierarchy-stamp.csv", ["file", "rows", "leaves", "sha256-16",
                                           "added-since-last-build", "removed-since-last-build"],
          [{"file": "records/cims-hierarchy.csv", "rows": len(h.nodes), "leaves": len(h.leaves),
            "sha256-16": h.digest,
            "added-since-last-build": "; ".join(added),
            "removed-since-last-build": "; ".join(removed)}], quiet=True)
    return {"added": added, "removed": removed, "first-build": not prev}
