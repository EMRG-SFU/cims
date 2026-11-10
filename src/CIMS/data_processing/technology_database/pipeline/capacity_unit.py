"""The unit each archetype's capacity is shown in (modeller 2026-10-28).

The pipeline holds capacity in one canonical unit per product unit (MW for GJ,
kt/yr for t, product-unit/yr otherwise; normalise.target_unit). Readers outside
the database expect each industry's own unit: t/d for an ammonia plant, Mt/yr for
a blast furnace, bbl/d for an oil sands mine, kW for a motor. The wide view shows
capacity in that unit, named in its own `capacity-unit` column, from the rules in
records/reference/capacity-unit.csv:

  * `match` is `archetype:<glob>`, `end-use:<name>` (the archetype's first end
    use), `product:<name>` or `product-unit:<unit>`; that is also the order of
    precedence; within a kind the first row that matches wins. One unit per end
    use and product (Brad 2026-10-28): units are set by product, with an end-use
    rule only where one product needs a different unit on one end use;
  * shown value = canonical value x `factor`, divided by the product's HHV
    (`energy-content`, GJ/t) and density (`density`, t/m3) where `divide-by`
    names them (records/reference/product-property.csv).

The long table (archetype-parameter.csv) and every calculation stay on the
canonical unit; only the wide views and the workbook change.
"""
from __future__ import annotations

from fnmatch import fnmatch

from .common import REFERENCE, read

RULES = REFERENCE / "capacity-unit.csv"
KINDS = ("archetype", "end-use", "product", "product-unit")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load():
    rules = {k: [] for k in KINDS}
    for r in read(RULES):
        kind, _, pat = r["match"].partition(":")
        if kind not in rules:
            raise SystemExit(f"capacity-unit.csv: match {r['match']!r} must start archetype:, end-use:, product: or product-unit:")
        rules[kind].append((pat, r))
    return rules


def rule_for(rules, row):
    home = (row.get("end-uses") or "").split(";")[0].strip()
    for kind, v in (("archetype", row.get("archetype", "")), ("end-use", home),
                    ("product", row.get("product", "")), ("product-unit", row.get("product-unit", ""))):
        for pat, r in rules[kind]:
            if (fnmatch(v, pat) if kind == "archetype" else v == pat):
                return r
    return None


def factor_for(rule, row):
    """Multiplier from the canonical unit to the rule's unit, or None when the
    product lacks a property the rule divides by."""
    f = _num(rule["factor"])
    for d in rule.get("divide-by", "").split():
        x = _num(row.get({"energy-content": "product-energy-content", "density": "product-density"}[d]))
        if not x:
            return None
        f /= x
    return f


def apply(rules, row, cols=("capacity", "capacity_low", "capacity_high")):
    """Rewrite the row's capacity cells in the shown unit; set `capacity-unit`.
    Returns the canonical unit kept when no rule applies (product property missing)."""
    rule = rule_for(rules, row)
    if _num(row.get("capacity")) is None:
        # no capacity (the no-abatement options): the unit is still shown, so the
        # column reads the same down an end use (Brad 2026-10-29)
        row["capacity-unit"] = rule["capacity-unit"] if rule else ""
        row["_capacity-canonical"] = {c: None for c in cols}
        return
    f = factor_for(rule, row) if rule else None
    if f is None:
        raise SystemExit(f"capacity-unit.csv: no usable rule for {row.get('archetype')} "
                         f"(product {row.get('product')!r}, unit {row.get('product-unit')!r})")
    row["capacity-unit"] = rule["capacity-unit"]
    # the canonical values, for the workbook bands that restate an archetype per GJ
    # of process heat (pipeline/bands.py)
    row["_capacity-canonical"] = {c: _num(row.get(c)) for c in cols}
    for c in cols:
        v = _num(row.get(c))
        if v is not None:
            row[c] = f"{float(f'{v * f:.5g}'):.12g}"   # 5 significant figures, no exponent
