"""Build the technology blocks included in the sector chapters.

Each block lists the archetypes from the technology database that compete
at one node of the CIMS hierarchy: their name, a one-line description,
and their type. The chapters pull
the blocks in with Quarto's include shortcode, for example::

    {{< include _technology-blocks/iron-and-steel/service_hot-metal.md >}}

so the lists stay in step with the database: rebuild after the database
changes and re-render the book.

Shared services (steam, machine drive, HVAC, lighting, and CO2 capture) are
documented once, in the General Services chapters. A sector's block for one
of those nodes lists only the archetypes specific to that sector and points
to the shared chapter for the rest.

Usage (from the repository root)::

    uv run --with pyyaml python docs/_technology-blocks/build_technology_blocks.py
    uv run --with pyyaml python docs/_technology-blocks/build_technology_blocks.py --coverage

``--coverage`` reports, for each sector, the hierarchy nodes with archetypes
whose block no chapter includes. The database location defaults to
``../../../../tech_data`` relative to this folder; set the environment
variable ``CIMS_TECH_DATA`` to point elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
DEFAULT_TECH_DATA = HERE / ".." / ".." / ".." / ".." / "tech_data"

TYPE_ORDER = ["reference", "conventional", "efficient variant", "alternative",
              "prospective", "component"]
TYPE_LABEL = {"reference": "Reference", "conventional": "Conventional",
              "efficient variant": "Efficiency variant", "alternative": "Alternative",
              "prospective": "Prospective", "component": "Component"}

# Shared services: the last part of the node name, the chapter that
# documents the shared archetypes, and that chapter's title.
SHARED = {
    "steam": ("industrial-heat.qmd", "Industrial Heat and Steam"),
    "pumps": ("machine-drive.qmd", "Industrial Machine Drive"),
    "fans-and-blowers": ("machine-drive.qmd", "Industrial Machine Drive"),
    "compressed-air": ("machine-drive.qmd", "Industrial Machine Drive"),
    "materials-handling": ("machine-drive.qmd", "Industrial Machine Drive"),
    "materials-processing": ("machine-drive.qmd", "Industrial Machine Drive"),
    "heating": ("hvac-and-lighting.qmd", "HVAC and Lighting"),
    "cooling": ("hvac-and-lighting.qmd", "HVAC and Lighting"),
    "lighting": ("hvac-and-lighting.qmd", "HVAC and Lighting"),
    "concentrated": ("ccus.qmd", "Carbon Capture, Utilization, and Storage"),
    "diffuse": ("ccus.qmd", "Carbon Capture, Utilization, and Storage"),
}

# Shared blocks for the General Services chapters: file name and the
# wildcard branch (or, for lighting, the archetype family) they list.
GENERAL = {
    "steam": "<sector>.*.service.steam",
    "machine-drive_pumps": "<sector>.*.service.machine-drive.pumps",
    "machine-drive_fans-and-blowers": "<sector>.*.service.machine-drive.fans-and-blowers",
    "machine-drive_compressed-air": "<sector>.*.service.machine-drive.compressed-air",
    "machine-drive_materials-handling": "<sector>.*.service.machine-drive.materials-handling",
    "machine-drive_materials-processing": "<sector>.*.service.machine-drive.materials-processing",
    "hvac_heating": "<sector>.*.service.hvac.heating",
    "hvac_cooling": "<sector>.*.service.hvac.cooling",
    "ccs_concentrated": "<sector>.*.service.ccs.concentrated",
    "ccs_diffuse": "<sector>.*.service.ccs.diffuse",
    "co2-transport_pipeline": "<root>.transmission.co2-pipeline-and-storage.*.pipeline",
    "co2-transport_storage": "<root>.transmission.co2-pipeline-and-storage.*.storage",
}


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def cell(text: str) -> str:
    return " ".join(text.replace("|", "\\|").split())


EDITS: dict = {}


def describe(arch: dict) -> str:
    """The archetype's description, with the edits in description_edits.yml applied."""
    text = EDITS.get("replace", {}).get(arch["archetype"], arch["description"])
    for pattern in EDITS.get("drop_sentences", []):
        text = re.sub(r"\s*" + pattern, "", text)
    return typeset(text)


def typeset(text: str) -> str:
    """Database shorthand to documentation style: degC to °C, CO2 to CO₂."""
    text = re.sub(r"\s?degC", " °C", text)
    return re.sub(r"\b(CO|N|NO|SO)2\b", lambda m: m.group(1) + "₂", text)


def sentence(text: str) -> str:
    text = cell(text)
    return text if not text or text.endswith((".", "!", "?")) else text + "."


def table(rows: list[dict]) -> list[str]:
    rows = sorted(rows, key=lambda r: (TYPE_ORDER.index(r["type"]) if r["type"] in TYPE_ORDER
                                        else len(TYPE_ORDER), r["label"].lower()))
    # dash counts set the relative column widths (Pandoc pipe tables)
    out = ["| Technology | What it is | Type |",
           "|:" + "-" * 26 + "|:" + "-" * 58 + "|:" + "-" * 16 + "|"]
    for r in rows:
        out.append(f"| {cell(typeset(r['label']))} | {sentence(describe(r))} | "
                   f"{TYPE_LABEL.get(r['type'], r['type'].capitalize())} |")
    return out


def block(rows: list[dict], heading: str, pointer: str = "") -> str:
    lines = ["::: {.tech-block}", heading, ""]
    if rows:
        lines += table(rows)
        lines.append("")
    if pointer:
        lines += [pointer, ""]
    lines.append(":::")
    return "\n".join(lines) + "\n"


def node_file(node: str) -> Path:
    """<sector>.iron-and-steel.service.hot-metal -> iron-and-steel/service_hot-metal.md"""
    parts = node.split(".")[1:]
    return Path(parts[0]) / ("_".join(parts[1:]) + ".md")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coverage", action="store_true",
                    help="report nodes whose block no chapter includes; write nothing")
    args = ap.parse_args(argv)

    global EDITS
    with open(HERE / "description_edits.yml", encoding="utf-8") as f:
        EDITS = yaml.safe_load(f) or {}
    root = Path(os.environ.get("CIMS_TECH_DATA", DEFAULT_TECH_DATA)).resolve()
    out_dir = root / "database" / "harmonized" / "output"
    archetypes = {r["archetype"]: r for r in read_csv(out_dir / "archetype.csv")}
    applications = read_csv(out_dir / "archetype-application.csv")

    by_node: dict[str, list[dict]] = defaultdict(list)
    by_wildcard: dict[str, dict[str, dict]] = defaultdict(dict)
    lighting: dict[str, dict] = {}
    for a in applications:
        arch = archetypes[a["archetype"]]
        if "*" in a["via"]:
            by_wildcard[a["via"]][a["archetype"]] = arch
        else:
            by_node[a["node"]].append(arch)
            if a["node"].endswith(".service.lighting"):
                lighting[a["archetype"]] = arch

    if args.coverage:
        included = set()
        for qmd in DOCS.glob("*.qmd"):
            included |= set(re.findall(r"_technology-blocks/([\w./-]+\.md)",
                                       qmd.read_text(encoding="utf-8")))
        missing = defaultdict(list)
        for node in sorted(by_node):
            if not node.startswith("<sector>."):
                continue
            if node.endswith(".service.lighting") and all(r["archetype"] in lighting for r in by_node[node]):
                continue
            if node_file(node).as_posix() not in included:
                missing[node.split(".")[1]].append(node)
        for sector, nodes in sorted(missing.items()):
            print(f"{sector}: {len(nodes)} node(s) without an included block")
            for n in nodes:
                print(f"  {n}")
        return 1 if missing else 0

    written = 0
    nodes = sorted(set(by_node) | {a["node"] for a in applications})
    for node in nodes:
        if not node.startswith("<sector>."):
            continue
        last = node.split(".")[-1]
        rows = by_node.get(node, [])
        pointer = ""
        if last in SHARED:
            qmd, title = SHARED[last]
            if last == "lighting":
                rows = [r for r in rows if r["archetype"] not in lighting]
            pointer = f"Shared technologies for this service are listed in [{title}]({qmd})."
        if not rows:
            continue                     # nothing sector-specific: the chapter links instead
        heading = "Technologies"
        path = HERE / node_file(node)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(block(rows, heading, pointer), encoding="utf-8")
        written += 1

    general = HERE / "general"
    general.mkdir(exist_ok=True)
    for name, via in GENERAL.items():
        rows = list(by_wildcard.get(via, {}).values())
        (general / f"{name}.md").write_text(block(rows, "Technologies"), encoding="utf-8")
        written += 1
    (general / "lighting.md").write_text(block(list(lighting.values()), "Technologies"),
                                         encoding="utf-8")
    written += 1
    print(f"wrote {written} technology blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
