"""Build the sector process flowcharts for the CIMS documentation.

Each sector spec in this folder (``<hierarchy-sector>.yml``) lists the
process steps, purchased inputs, site services, and products of one sector,
the pictogram for each, and the flows between them. This script lays them
out on a grid, applies the shared style in ``flowchart_style.yml``, and
writes one light and one dark SVG per sector to ``images/flowcharts/``.

It also checks every ``branch:`` in a spec against the CIMS hierarchy
(``cims_hierarchy.csv``), so a chart fails to build when it names a node the
model no longer has.

Usage (from the repository root)::

    uv run --with pyyaml python docs/flowcharts/build_flowcharts.py
    uv run --with pyyaml python docs/flowcharts/build_flowcharts.py iron-and-steel
    uv run --with pyyaml python docs/flowcharts/build_flowcharts.py --check

The appendix "Figure style" (appendix_figure-style.qmd) describes the spec
format and the style settings.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import yaml

HERE = Path(__file__).resolve().parent
STYLE_FILE = HERE / "flowchart_style.yml"
ICON_DIR = HERE / "icons"
NON_SPEC_FILES = {"flowchart_style.yml"}
MODES = ("light", "dark")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_icon(name: str) -> str:
    """Return the inner markup of icons/<name>.svg wrapped as a <symbol>."""
    path = ICON_DIR / f"{name}.svg"
    if not path.exists():
        raise SystemExit(f"Unknown icon '{name}': {path} does not exist")
    text = path.read_text(encoding="utf-8")
    m = re.search(r"<svg([^>]*)>(.*)</svg>", text, re.S)
    if not m:
        raise SystemExit(f"Cannot parse icon {path}")
    attrs = dict(re.findall(r'([\w:-]+)="([^"]*)"', m.group(1)))
    keep = ("fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin")
    g_attrs = " ".join(f'{k}="{attrs[k]}"' for k in keep if k in attrs)
    view_box = attrs.get("viewBox", "0 0 48 48")
    return (f'<symbol id="icon-{name}" viewBox="{view_box}">'
            f'<g {g_attrs}>{m.group(2).strip()}</g></symbol>')


def load_hierarchy(style: dict) -> set[str] | None:
    path = os.environ.get("CIMS_HIERARCHY_CSV") or (HERE / style["hierarchy_csv"])
    path = Path(path)
    if not path.exists():
        print(f"warning: hierarchy not found at {path}; branch check skipped",
              file=sys.stderr)
        return None
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row["branch"] for row in csv.DictReader(f)}


def full_branch(branch: str) -> str:
    if branch.startswith("<"):
        return branch
    return f"<sector>.{branch}"


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

@dataclass
class Node:
    id: str
    label: str
    role: str
    icon: str
    col: float
    row: float
    note: str = ""
    branches: list[str] = field(default_factory=list)
    cx: float = 0.0
    cy: float = 0.0


@dataclass
class Edge:
    src: str
    dst: str
    kind: str = "material"
    label: str = ""
    route: str = "auto"      # auto, below, above
    turn: str = "mid"        # where a Z-shaped route turns: early, mid, late
    sides: tuple = ()        # explicit (source side, target side)
    src_side: str = ""
    dst_side: str = ""
    src_off: float = 0.0
    dst_off: float = 0.0


def wrap(text: str, max_chars: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def rounded_path(points: list[tuple[float, float]], r: float) -> str:
    """Polyline through points with rounded corners of radius r."""
    if len(points) < 3:
        return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)
    d = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
    for i in range(1, len(points) - 1):
        (x0, y0), (x1, y1), (x2, y2) = points[i - 1], points[i], points[i + 1]
        l1 = max(abs(x1 - x0), abs(y1 - y0))
        l2 = max(abs(x2 - x1), abs(y2 - y1))
        rr = min(r, l1 / 2, l2 / 2)
        if rr < 0.5:
            d.append(f"L {x1:.1f} {y1:.1f}")
            continue
        ax = x1 - (x1 - x0) / l1 * rr if l1 else x1
        ay = y1 - (y1 - y0) / l1 * rr if l1 else y1
        bx = x1 + (x2 - x1) / l2 * rr if l2 else x1
        by = y1 + (y2 - y1) / l2 * rr if l2 else y1
        d.append(f"L {ax:.1f} {ay:.1f} Q {x1:.1f} {y1:.1f} {bx:.1f} {by:.1f}")
    d.append(f"L {points[-1][0]:.1f} {points[-1][1]:.1f}")
    return " ".join(d)


class Chart:
    def __init__(self, spec: dict, style: dict):
        self.spec = spec
        self.style = style
        self.sector = spec["sector"]
        lay, nd = style["layout"], style["node"]
        self.w, self.h = nd["width"], nd["height"]
        self.groups = spec.get("groups", [])
        self.top_extra = lay["group_title_height"] + lay["group_pad"] if self.groups else 0
        self.nodes: dict[str, Node] = {}
        for n in spec["nodes"]:
            col, row = n["at"]
            branches = n.get("branches") or ([n["branch"]] if n.get("branch") else [])
            node = Node(id=n["id"], label=n["label"], role=n.get("role", "process"),
                        icon=n.get("icon", "process"), col=col, row=row,
                        note=n.get("note", ""), branches=branches)
            node.cx = lay["margin"] + col * lay["col_width"] + lay["col_width"] / 2
            node.cy = (lay["margin"] + self.top_extra + row * lay["row_height"]
                       + lay["row_height"] / 2)
            if node.id in self.nodes:
                raise SystemExit(f"{self.sector}: duplicate node id '{node.id}'")
            self.nodes[node.id] = node
        self.edges: list[Edge] = []
        for e in spec.get("edges", []):
            if isinstance(e, list):
                e = {"from": e[0], "to": e[1], "kind": e[2] if len(e) > 2 else "material"}
            edge = Edge(src=e["from"], dst=e["to"], kind=e.get("kind", "material"),
                        label=e.get("label", ""), route=e.get("route", "auto"),
                        turn=e.get("turn", "mid"), sides=tuple(e.get("sides", ())))
            for end in (edge.src, edge.dst):
                if end not in self.nodes:
                    raise SystemExit(f"{self.sector}: edge refers to unknown node '{end}'")
            self.edges.append(edge)
        self._assign_ports()

    # ---- ports: which side each edge leaves and enters, spread along the side
    def _assign_ports(self) -> None:
        eps = 1.0
        for e in self.edges:
            s, t = self.nodes[e.src], self.nodes[e.dst]
            if e.sides:
                e.src_side, e.dst_side = e.sides
            elif e.route == "below":
                e.src_side, e.dst_side = "bottom", "bottom"
            elif e.route == "above":
                e.src_side, e.dst_side = "top", "top"
            elif t.cx > s.cx + eps:
                e.src_side, e.dst_side = "right", "left"
            elif t.cx < s.cx - eps:
                e.src_side, e.dst_side = "bottom", "bottom"
            elif t.cy > s.cy:
                e.src_side, e.dst_side = "bottom", "top"
            else:
                e.src_side, e.dst_side = "top", "bottom"
        slots: dict[tuple[str, str], list[tuple[float, Edge, str]]] = {}
        for e in self.edges:
            s, t = self.nodes[e.src], self.nodes[e.dst]
            key_s = (e.src, e.src_side)
            key_t = (e.dst, e.dst_side)
            along_s = t.cy if e.src_side in ("left", "right") else t.cx
            along_t = s.cy if e.dst_side in ("left", "right") else s.cx
            slots.setdefault(key_s, []).append((along_s, e, "src"))
            slots.setdefault(key_t, []).append((along_t, e, "dst"))
        for (nid, side), items in slots.items():
            items.sort(key=lambda it: it[0])
            n = len(items)
            span = (self.h if side in ("left", "right") else self.w) - 28
            step = min(12.0, span / max(n - 1, 1)) if n > 1 else 0
            for i, (_, e, end) in enumerate(items):
                off = (i - (n - 1) / 2) * step
                if end == "src":
                    e.src_off = off
                else:
                    e.dst_off = off
        # straight runs: when one end is alone on its side, line it up with
        # the other end so the edge stays straight instead of jogging
        counts = {k: len(v) for k, v in slots.items()}
        for e in self.edges:
            s, t = self.nodes[e.src], self.nodes[e.dst]
            horiz = {e.src_side, e.dst_side} == {"left", "right"}
            vert = {e.src_side, e.dst_side} == {"top", "bottom"}
            if horiz:
                gap = (t.cy + e.dst_off) - s.cy
                if counts[(e.src, e.src_side)] == 1 and abs(gap) < self.h / 2 - 12:
                    e.src_off = gap
                gap = (s.cy + e.src_off) - t.cy
                if counts[(e.dst, e.dst_side)] == 1 and abs(gap) < self.h / 2 - 12:
                    e.dst_off = gap
            elif vert:
                gap = (t.cx + e.dst_off) - s.cx
                if counts[(e.src, e.src_side)] == 1 and abs(gap) < self.w / 2 - 12:
                    e.src_off = gap
                gap = (s.cx + e.src_off) - t.cx
                if counts[(e.dst, e.dst_side)] == 1 and abs(gap) < self.w / 2 - 12:
                    e.dst_off = gap

    def port(self, node: Node, side: str, off: float) -> tuple[float, float]:
        if side == "right":
            return node.cx + self.w / 2, node.cy + off
        if side == "left":
            return node.cx - self.w / 2, node.cy + off
        if side == "top":
            return node.cx + off, node.cy - self.h / 2
        return node.cx + off, node.cy + self.h / 2

    def route(self, e: Edge) -> list[tuple[float, float]]:
        lay = self.style["layout"]
        s, t = self.nodes[e.src], self.nodes[e.dst]
        p0 = self.port(s, e.src_side, e.src_off)
        p1 = self.port(t, e.dst_side, e.dst_off)
        horizontal = {"left", "right"}
        if e.src_side in horizontal and e.dst_side in horizontal and e.src_side != e.dst_side:
            if abs(p0[1] - p1[1]) < 0.5:
                return [p0, p1]
            sign = 1 if p1[0] > p0[0] else -1
            gap = lay["col_width"] - self.w          # free space between columns
            if e.turn == "late":
                inset = min(max(gap / 2 - e.dst_off * 0.5, 5), gap - 5)
                mx = p1[0] - sign * inset
            elif e.turn == "early":
                inset = min(max(gap / 2 + e.src_off * 0.5, 5), gap - 5)
                mx = p0[0] + sign * inset
            else:
                mx = (p0[0] + p1[0]) / 2
            return [p0, (mx, p0[1]), (mx, p1[1]), p1]
        if (e.src_side in horizontal) != (e.dst_side in horizontal):
            # one horizontal and one vertical side: a single corner
            if e.src_side in horizontal:
                return [p0, (p1[0], p0[1]), p1]
            return [p0, (p0[0], p1[1]), p1]
        if e.src_side == "bottom" and e.dst_side == "bottom":
            y = max(s.cy, t.cy) + self.h / 2 + (lay["row_height"] - self.h) / 2
            return [p0, (p0[0], y), (p1[0], y), p1]
        if e.src_side == "top" and e.dst_side == "top":
            y = min(s.cy, t.cy) - self.h / 2 - (lay["row_height"] - self.h) / 2
            return [p0, (p0[0], y), (p1[0], y), p1]
        if abs(p0[0] - p1[0]) < 0.5:
            return [p0, p1]
        my = (p0[1] + p1[1]) / 2
        return [p0, (p0[0], my), (p1[0], my), p1]

    # ---- rendering
    def render(self, mode: str, legend: bool = True) -> str:
        st = self.style
        pal = st["palettes"][mode]
        font, lay, nd, ed = st["font"], st["layout"], st["node"], st["edge"]
        out: list[str] = []
        icons_used = sorted({n.icon for n in self.nodes.values()})

        max_col = max(n.col for n in self.nodes.values())
        max_row = max(n.row for n in self.nodes.values())
        width = lay["margin"] * 2 + (max_col + 1) * lay["col_width"]
        body_h = lay["margin"] + self.top_extra + (max_row + 1) * lay["row_height"]

        # groups
        for g in self.groups:
            c0, c1 = g["cols"]
            r0, r1 = g["rows"]
            x0 = lay["margin"] + c0 * lay["col_width"] + lay["col_width"] / 2 - self.w / 2 - lay["group_pad"]
            x1 = lay["margin"] + c1 * lay["col_width"] + lay["col_width"] / 2 + self.w / 2 + lay["group_pad"]
            y0 = (lay["margin"] + self.top_extra + r0 * lay["row_height"] + lay["row_height"] / 2
                  - self.h / 2 - lay["group_pad"] - lay["group_title_height"])
            y1 = (lay["margin"] + self.top_extra + r1 * lay["row_height"] + lay["row_height"] / 2
                  + self.h / 2 + lay["group_pad"])
            out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" '
                       f'rx="{nd["radius"] + 4}" fill="{pal["group_fill"]}" stroke="{pal["group_stroke"]}"/>')
            out.append(f'<text x="{x0 + 10:.1f}" y="{y0 + lay["group_title_height"] - 6:.1f}" '
                       f'font-size="{font["group_size"]}" font-weight="{font["group_weight"]}" '
                       f'letter-spacing="{font["group_letter_spacing"]}" fill="{pal["group_text"]}">'
                       f'{escape(g["label"].upper())}</text>')

        # edges (under the cards)
        kinds_used = []
        for e in self.edges:
            kind = ed["kinds"][e.kind]
            colour = pal[kind["colour"]]
            pts = self.route(e)
            dash = f' stroke-dasharray="{kind["dash"]}"' if kind["dash"] else ""
            out.append(f'<path d="{rounded_path(pts, ed["corner_radius"])}" fill="none" '
                       f'stroke="{colour}" stroke-width="{ed["width"]}"{dash} '
                       f'marker-end="url(#arrow-{e.kind})"/>')
            if e.kind not in kinds_used:
                kinds_used.append(e.kind)
            if e.label:
                # label on the longest segment
                segs = list(zip(pts[:-1], pts[1:]))
                (ax, ay), (bx, by) = max(segs, key=lambda s: abs(s[1][0] - s[0][0]) + abs(s[1][1] - s[0][1]))
                if abs(bx - ax) >= abs(by - ay):      # horizontal: centre the label above
                    lx, ly, anchor = (ax + bx) / 2, (ay + by) / 2 - ed["label_offset"], "middle"
                else:                                 # vertical: set it beside the line
                    lx, ly, anchor = ax + ed["label_offset"] + 2, (ay + by) / 2 + 4, "start"
                out.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
                           f'font-size="{font["note_size"]}" fill="{pal["muted"]}" '
                           f'stroke="{pal.get("halo", "none")}" stroke-width="4" '
                           f'paint-order="stroke">{escape(e.label)}</text>')

        # nodes
        roles_used = []
        for n in self.nodes.values():
            role = st["roles"][n.role]
            if n.role not in roles_used:
                roles_used.append(n.role)
            x, y = n.cx - self.w / 2, n.cy - self.h / 2
            dash = f' stroke-dasharray="{role["dash"]}"' if role.get("dash") else ""
            tip = n.label + (" — " + ", ".join(n.branches) if n.branches else "")
            out.append(f'<g><title>{escape(tip)}</title>')
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{self.w}" height="{self.h}" '
                       f'rx="{nd["radius"]}" fill="{pal[role["fill"]]}" stroke="{pal[role["stroke"]]}" '
                       f'stroke-width="{nd["stroke_width"]}"{dash}/>')
            isz = nd["icon_size"]
            out.append(f'<use href="#icon-{n.icon}" x="{n.cx - isz / 2:.1f}" y="{y + nd["icon_top"]:.1f}" '
                       f'width="{isz}" height="{isz}" style="color:{pal[role["icon"]]}"/>')
            lines = wrap(n.label, nd["label_max_chars"])
            lh = font["label_size"] * nd["label_line_height"]
            ty = y + nd["icon_top"] + isz + 4 + font["label_size"]
            if len(lines) > 3:
                print(f"warning: {self.sector}: label '{n.label}' needs more than three lines",
                      file=sys.stderr)
            lines = lines[:3]
            ty += lh / 2 * (2 - len(lines))   # centre one, two, or three lines in the label area
            for i, line in enumerate(lines):
                out.append(f'<text x="{n.cx:.1f}" y="{ty + i * lh:.1f}" text-anchor="middle" '
                           f'font-size="{font["label_size"]}" font-weight="{font["label_weight"]}" '
                           f'fill="{pal[role["text"]]}">{escape(line)}</text>')
            if n.note:
                out.append(f'<text x="{n.cx:.1f}" y="{y + self.h + font["note_size"] + 3:.1f}" '
                           f'text-anchor="middle" font-size="{font["note_size"]}" '
                           f'fill="{pal["muted"]}">{escape(n.note)}</text>')
            out.append("</g>")

        # legend
        height = body_h + lay["margin"]
        if legend:
            fs = font["note_size"]
            items = []                      # (kind, key, text)
            for r in st["roles"]:
                if r in roles_used:
                    items.append(("role", r, st["roles"][r]["legend"]))
            for k in st["legend_edges"]:
                if k in kinds_used:
                    items.append(("edge", k, st["legend_edges"][k]))
            x0 = lay["margin"] + 4
            lx, ly = x0, body_h + lay["legend_gap"]
            for kind, key, text in items:
                swatch = 22 if kind == "role" else 36
                item_w = swatch + len(text) * fs * font["char_width"] + 22
                if lx > x0 and lx + item_w > width - lay["margin"]:
                    lx, ly = x0, ly + fs * 2
                if kind == "role":
                    role = st["roles"][key]
                    dash = f' stroke-dasharray="{role["dash"]}"' if role.get("dash") else ""
                    out.append(f'<rect x="{lx:.1f}" y="{ly - 10}" width="16" height="12" rx="3" '
                               f'fill="{pal[role["fill"]]}" stroke="{pal[role["stroke"]]}"{dash}/>')
                else:
                    ek = ed["kinds"][key]
                    dash = f' stroke-dasharray="{ek["dash"]}"' if ek["dash"] else ""
                    out.append(f'<path d="M {lx:.1f} {ly - 4} H {lx + 28:.1f}" stroke="{pal[ek["colour"]]}" '
                               f'stroke-width="{ed["width"]}"{dash} marker-end="url(#arrow-{key})"/>')
                out.append(f'<text x="{lx + swatch:.1f}" y="{ly}" font-size="{fs}" fill="{pal["muted"]}">'
                           f'{escape(text)}</text>')
                lx += item_w
            height = ly + lay["margin"]

        defs = [load_icon(i) for i in icons_used]
        a = ed["arrow_size"]
        for k, kind in ed["kinds"].items():
            defs.append(f'<marker id="arrow-{k}" viewBox="0 0 10 10" refX="9" refY="5" '
                        f'markerWidth="{a}" markerHeight="{a}" markerUnits="userSpaceOnUse" orient="auto-start-reverse">'
                        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{pal[kind["colour"]]}"/></marker>')
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
                f'width="{width:.0f}" height="{height:.0f}" font-family="{escape(font["family"], {chr(34): "&quot;"})}" '
                f'role="img" aria-label="{escape(self.spec.get("title", self.sector))} process flowchart">')
        return "\n".join([head, "<defs>", *defs, "</defs>", *out, "</svg>", ""])

    def check(self, hierarchy: set[str] | None) -> list[str]:
        if hierarchy is None:
            return []
        problems = []
        for n in self.nodes.values():
            for b in n.branches:
                if full_branch(b) not in hierarchy:
                    problems.append(f"{self.sector}: node '{n.id}' names branch '{b}', "
                                    f"which is not in the hierarchy")
        return problems


# --------------------------------------------------------------------------
# Icon gallery (for the appendix)
# --------------------------------------------------------------------------

def icon_gallery(style: dict, mode: str, per_row: int = 10) -> str:
    pal = style["palettes"][mode]
    font = style["font"]
    names = sorted(p.stem for p in ICON_DIR.glob("*.svg"))
    cell_w, cell_h, isz = 92, 82, 36
    rows = (len(names) + per_row - 1) // per_row
    width, height = per_row * cell_w + 16, rows * cell_h + 16
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
           f'height="{height}" font-family="{escape(font["family"], {chr(34): "&quot;"})}" role="img" '
           f'aria-label="Flowchart icon gallery">', "<defs>", *[load_icon(n) for n in names], "</defs>"]
    for i, name in enumerate(names):
        cx = 8 + (i % per_row) * cell_w + cell_w / 2
        y = 8 + (i // per_row) * cell_h
        out.append(f'<use href="#icon-{name}" x="{cx - isz / 2}" y="{y + 8}" width="{isz}" height="{isz}" '
                   f'style="color:{pal["process_icon"]}"/>')
        out.append(f'<text x="{cx}" y="{y + isz + 26}" text-anchor="middle" font-size="10.5" '
                   f'fill="{pal["muted"]}">{escape(name)}</text>')
    out.append("</svg>\n")
    return "\n".join(out)


def style_sample(style: dict, mode: str) -> str:
    """One card for each role and one line for each kind of flow, for the appendix."""
    r = style["roles"]
    spec = {"sector": "style-sample", "title": "Style sample", "nodes": [
        {"id": "input", "label": r["input"]["legend"], "icon": "ore", "role": "input", "at": [0, 1]},
        {"id": "process", "label": r["process"]["legend"], "icon": "furnace", "role": "process", "at": [1, 1]},
        {"id": "output", "label": r["output"]["legend"], "icon": "coil", "role": "output", "at": [2, 1]},
        {"id": "service", "label": r["service"]["legend"], "icon": "boiler", "role": "service", "at": [1, 2]},
        {"id": "emission", "label": r["emission"]["legend"], "icon": "ccs", "role": "emission", "at": [1, 0]},
    ], "edges": [
        {"from": "input", "to": "process"},
        {"from": "process", "to": "output"},
        {"from": "service", "to": "process", "kind": "energy"},
        {"from": "process", "to": "emission", "kind": "emission"},
    ]}
    return Chart(spec, style).render(mode, legend=False)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sectors", nargs="*", help="spec names to build (default: all)")
    ap.add_argument("--check", action="store_true", help="check branches only; write nothing")
    args = ap.parse_args(argv)

    style = load_yaml(STYLE_FILE)
    hierarchy = load_hierarchy(style)
    out_dir = (HERE / style["output_dir"]).resolve()
    specs = sorted(p for p in HERE.glob("*.yml") if p.name not in NON_SPEC_FILES)
    if args.sectors:
        wanted = {s.removesuffix(".yml") for s in args.sectors}
        specs = [p for p in specs if p.stem in wanted]
        missing = wanted - {p.stem for p in specs}
        if missing:
            raise SystemExit(f"No spec for: {', '.join(sorted(missing))}")

    problems: list[str] = []
    charts = []
    for path in specs:
        spec = load_yaml(path)
        spec.setdefault("sector", path.stem)
        chart = Chart(spec, style)
        problems += chart.check(hierarchy)
        charts.append(chart)
    for p in problems:
        print("error:", p, file=sys.stderr)
    if problems:
        return 1
    if args.check:
        print(f"{len(charts)} specs checked; all branches found")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for chart in charts:
        for mode in MODES:
            name = style["output_name"].format(sector=chart.sector, mode=mode)
            (out_dir / name).write_text(chart.render(mode), encoding="utf-8")
        print(f"built {chart.sector}")
    if not args.sectors:
        for mode in MODES:
            (out_dir / f"icon-gallery_{mode}.svg").write_text(icon_gallery(style, mode), encoding="utf-8")
            (out_dir / f"style-sample_{mode}.svg").write_text(style_sample(style, mode), encoding="utf-8")
        print("built icon gallery and style sample")
    return 0


if __name__ == "__main__":
    sys.exit(main())
