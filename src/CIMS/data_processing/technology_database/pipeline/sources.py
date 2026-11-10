"""The source list and the citation review.

source-list.csv holds every source behind the database: short form, full
citation, role, whether the document is held locally, how many fact rows it
contributes and which families it informs. Full citations come from, in order:
records/reference/source-citation.csv (the verified record, with a DOI only where
a fetched page showed one); the References tab of the UKERC workbook, for the
originals UKERC cites; the author, title and year in the source manifest; and
otherwise nothing -- the row says so rather than inventing one.

citation-review.md reports what the citation check found: corrections, merges,
renames, and what is still unresolved.
"""
from __future__ import annotations

import collections
import csv
import re
import warnings

from .common import OUTPUT, RAW, REFERENCE, read, write

warnings.filterwarnings("ignore")
UKERC_REFS = RAW / "other-reports" / "ukerc_2013_usable-energy-database_references.csv"
MANIFEST = RAW / "source-details_sources.csv"
MANIFEST_FILES = RAW / "source-details_files.csv"
CITES = RAW / "other-reports" / "ukerc_2013_usable-energy-database_citations.csv"
VERIFIED = REFERENCE / "source-citation.csv"

# Same document, spelled differently in the data tables and the References tab.
# Each is a transposition, a dropped letter or a missing 'et al'; none changes
# which document is meant.
ALIAS = {
    "IISI 1996": "ISII 1996",                       # tab transposes IISI
    "Stemmermann 2010": "Stemmermann et al. 2010",
    "Von Weizsacker 2009": "Von Weizsacker et al. 2009",
    "Duxon 2007": "Duxon and Provis 2007",
    "IIP 2012": "ITP 2012",                         # tab uses the older acronym
    "Collodi 2010": "Collodi and Wheeler 2010",     # tab keeps the mis-parsed author
}
# Full citations supplied by the modeller 2026-08-28 where the References tab
# could not resolve the short form. Recorded here, with their provenance, rather
# than left to a later reader to rediscover.
OVERRIDE = {
    "EC 2012": (
        "Joint Research Centre: Institute for Prospective Technological Studies, "
        "Aguado-Monsonet, M. A., Delgado Sancho, L., Remus, R. and Roudier, S., "
        "Best available techniques (BAT) reference document for iron and steel "
        "production - Industrial emissions Directive 2010/75/EU - Integrated "
        "pollution prevention and control, Publications Office, 2013, "
        "https://data.europa.eu/doi/10.2791/98516",
        "supplied by the modeller for the short form IPPC 2012, which was merged into EC 2012 on 2026-08-29; the References tab's nearest entry is the 2010 draft of the same BAT reference document"),
}
UNRESOLVED = {}
# "IEAGHG 2013" (no suffix) was resolved 2026-11-04: it is IEAGHG 2013a (2013/TR3); see MERGED.

# Ids left behind when a short form was canonicalized. They stay registered in
# the manifest (it is append-only and the history is worth keeping) but they are
# superseded, not pending.
SUPERSEDED = {
    "itp_2012_001": "iip_2012_001",
    # ietd_2013_001 was retired here 2026-08-29 but came back 2026-10-07 as the held IETD
    # archive, so it is live again (2026-11-04)

    "fritzson_2006_002": "fritzson-berntsson_2006_002",
    "fritzson_2006_003": "fritzson-berntsson_2006_003",
    "novacem_2011_002": "novacem_2011_001",
    # merged 2026-08-29 after the citation check confirmed each pair is one document
    "martin_2000_003": "martin_2000_001",
    "ieaghg_2000_001": "ieaghg_2013_002",   # merged 2026-11-04 (user): one report, 2013/TR3
    "ieaghg_2013_001": "ieaghg_2013_002",
    "ecra_2009_002": "ecra_2009_001",
    "novacem_2011_003": "novacem_2011_001",
    "ippc_2012_001": "ec_2012_001",
    # Renamed 2026-08-29: the issuer was misidentified, not the document.
    "collodi-wheeler_2010_001": "collodi_2010_001",
    # the held file is the 2024 edition, not the 2026 one it was registered as
    "iea_2026_002": "iea_2024_001",  # the Electrification report that reused this id is iea_2026_006 (2026-11-04)
    "nera-ea_2009_001": "nera-aea_2009_001",
    "nera_2009_001": "nera-aea_2009_001",
    "ecra_2009_003": "ecra_2009_002",
    "normal_2013_001": "norman_2013_001",
}


def key(c):
    c = re.sub(r"\s+", " ", str(c)).strip().rstrip(".")
    return c.replace("et al.", "et al").replace("IEA GHG", "IEAGHG").lower()


def ukerc_references():
    """UKERC's References tab, extracted to ..._references.csv (sector, citation as
    UKERC writes it, full reference), so the build needs no workbook (2026-11-06)."""
    if not UKERC_REFS.exists():
        return {}
    out = {}
    for r in read(UKERC_REFS):
        out.setdefault(key(r["citation"]), (r["reference"], r["sector"]))
    return out


def manifest():
    src = {}
    if MANIFEST.exists():
        for d in csv.DictReader(MANIFEST.open(encoding="utf-8-sig")):
            if d.get("source_id"):
                src[d["source_id"]] = d
    files = collections.defaultdict(list)
    if MANIFEST_FILES.exists():
        for d in csv.DictReader(MANIFEST_FILES.open(encoding="utf-8-sig")):
            # A row with no file on disk is a registration, not a holding, and
            # must not make the source read as held locally.
            if d.get("source_id") and d.get("file_on_disk"):
                files[d["source_id"]].append(d["file_name"])
    return src, files


def compose(v: dict) -> str:
    """Author. Year. Title. Publisher. -- one house style for every source, so
    the list reads as one document rather than as UKERC's formatting plus ours.
    The DOI rides in its own column, not glued onto the end of the string."""
    bits = [v.get("verified-authors", ""), v.get("verified-year", ""),
            v.get("verified-title", ""), v.get("verified-publisher", "")]
    out = ". ".join(b.strip().rstrip(".") for b in bits if b.strip())
    return out + "." if out else ""


# Findings that change more than a citation string: a wrong source-id, a pair of
# ids that turn out to be one document, a report code that does not exist. These
# are decisions for the modeller, not edits this script should make silently.
STRUCTURAL = [
    ("etsap_2013_002", "the web check reported the IEA-ETSAP/IRENA heat pump brief as E12. "
     "The copy held in data/raw_data/technology_database/iea/etsap prints 'Technology Policy Brief E19 - January "
     "2013' on page 1, and the modeller confirmed E19. The record follows the document; "
     "IRENA's website appears to post the same brief under E12."),
]

# Merged on the modeller's instruction 2026-08-29, once the citation check had
# confirmed each pair is one document. The retired short form is rewritten in
# ukerc-citations.py, so the rows move with it; the retired id stays registered
# in the manifest and is labelled superseded.
MERGED = [
    ("Martin et al 2000b -> Martin et al 2000", "martin_2000_003 -> martin_2000_001",
     "one report, LBNL-46990 / ACEEE IE003"),
    ("ECRA 2009a -> ECRA 2009", "ecra_2009_002 -> ecra_2009_001",
     "the References tab has a single ECRA 2009 entry"),
    ("Novacem 2011b -> Novacem 2011", "novacem_2011_003 -> novacem_2011_001",
     "one 31st Cement and Concrete Science Conference paper"),
    ("IPPC 2012 -> EC 2012", "ippc_2012_001 -> ec_2012_001",
     "one iron and steel BREF; 'EC' names the issuer where 'IPPC' names a directive"),
    ("IEAGHG 2000, IEAGHG 2013 -> IEAGHG 2013a", "ieaghg_2000_001, ieaghg_2013_001 -> ieaghg_2013_002",
     "one report, IEAGHG 2013/TR3 (merged 2026-11-04 on the user's instruction): the References tab "
     "lists it as 2013a, the data tables cite it unsuffixed and, misdated, as 2000; the modeller had "
     "identified the 2000 form as 2013-TR3. TR3 prints no costs, so UKERC's cost rows on it are UKERC's own"),
]

# Renames, not merges: the document is the same, the attribution was wrong. It
# was applied 2026-08-29. The retired id is superseded, not deleted, and the
# replacement is registered in the manifest with the reason.
RENAMED = [
    ("collodi-wheeler_2010_001 -> collodi_2010_001", "UKERC's 'Wheeler, F.' is the "
     "affiliation Foster Wheeler mis-parsed as a second author; Collodi is sole author"),
]




# short name first, so the sheet reads by name (Brad 2026-10-29)
COLS = ["short-name", "source-id", "full-citation", "year", "doi", "url", "verification",
        "role", "held-locally", "rows-in-database", "families", "licence",
        "citation-as-cited", "note"]


def source_list(records, examples, fact_tables) -> list[dict]:
    refs = ukerc_references()
    verified = {r["source-id"]: r for r in read(VERIFIED)}
    src, files = manifest()
    cites = {r["source-id"]: r for r in read(CITES)
             if r["source-id"] and r["source-id"] != "unattributed"}
    fam = {a["archetype"]: a["family"] for a in records.archetypes}
    used, vias, families = collections.Counter(), collections.Counter(), collections.defaultdict(set)
    through = {}
    for e in examples.values():
        # a source whose only contribution is a readiness level on an example
        # (the IEA guide) still counts as used
        if e.get("archetype"):
            families[e["source-id"]].add(fam.get(e["archetype"], ""))
    for rows in fact_tables:
        for r in rows:
            sid = (r.get("source-id") or "").strip()
            if not sid:
                continue
            used[sid] += 1
            if r.get("archetype"):
                families[sid].add(fam.get(r["archetype"], ""))
            if r.get("via-source-id"):
                vias[r["via-source-id"]] += 1
                through.setdefault(sid, r["via-source-id"])
    for e in examples.values():
        if not used.get(e["source-id"]) and not vias.get(e["source-id"]):
            used[e["source-id"]] += 0
    rows, unmatched = [], []
    for sid in sorted(set(used) | set(vias) | set(src)):
        m = src.get(sid, {})
        short = records.priority.get(sid, {}).get("short-name") or m.get("author") or sid
        cite_short = cites.get(sid, {}).get("citation", "")
        full, note = "", ""
        if cite_short and cite_short in OVERRIDE:
            full, note = OVERRIDE[cite_short]
        elif cite_short:
            hit = refs.get(key(cite_short)) or refs.get(key(ALIAS.get(cite_short, "")))
            if hit:
                full = hit[0]
                if cite_short in ALIAS:
                    note = (f"the References tab spells this {ALIAS[cite_short]!r} where the "
                            f"data tables spell it {cite_short!r}")
            else:
                note = UNRESOLVED.get(cite_short, "not found in the UKERC References tab")
                unmatched.append(cite_short)
        stub = "cited by the UKERC Usable Energy Database" in m.get("title", "")
        if not full and m and not stub:
            bits = [m.get("author", ""), m.get("year", ""), m.get("title", "")]
            inst = m.get("institution", "")
            if inst and inst != m.get("author"):
                bits.append(inst)
            full = ". ".join(b for b in bits if b)
        fams = "; ".join(sorted(f for f in families.get(sid, ()) if f))
        if sid in SUPERSEDED:
            rows.append({"source-id": sid, "short-name": short, "year": m.get("year", ""),
                         "role": "superseded", "held-locally": "no", "rows-in-database": 0,
                         "note": f"superseded by {SUPERSEDED[sid]}"})
            continue
        via_of = through.get(sid)
        role = ("cited by UKERC" if cite_short else
                "compilation" if vias.get(sid) else
                f"cited by {records.short(via_of)}" if via_of else
                "extracted directly" if sid in used else "registered, not yet used")
        v = verified.get(sid, {})
        as_cited = ""
        if v and compose(v):
            as_cited, full = full, compose(v)
            if v.get("verification-note"):
                note = "; ".join(x for x in (note, v["verification-note"]) if x)
        year = v.get("verified-year") or m.get("year", "") or (sid.split("_")[1] if "_" in sid else "")
        rows.append({"source-id": sid, "short-name": short, "full-citation": full, "year": year,
                     "doi": v.get("doi", ""), "url": v.get("url", ""),
                     "verification": v.get("verification-status", "not checked"),
                     "citation-as-cited": as_cited, "role": role,
                     "held-locally": "yes" if files.get(sid) else "no",
                     "rows-in-database": used.get(sid, 0) or vias.get(sid, 0),
                     "families": fams, "licence": m.get("licence_type", ""), "note": note})
    for r in rows:
        if not r.get("full-citation") and not r.get("note"):
            r["note"] = "no full citation available"
    # a source with no family gets the reason in its note
    for r in rows:
        if r.get("families"):
            continue
        n = int(r.get("rows-in-database") or 0)
        if r.get("role") == "superseded":
            why = "no family: superseded, so nothing from it is loaded"
        elif n == 0 and r.get("held-locally") == "yes":
            why = ("no family: held, but nothing from it is loaded yet (queued for extraction, "
                   "or registered as background/documentation)")
        elif n == 0 and r.get("role") == "cited by UKERC":
            why = "no family: cited in the UKERC reference list, but no UKERC row that reaches the database cites it"
        elif n == 0:
            why = "no family: registered but not held and not yet used in the database"
        else:
            why = (f"no family: its {n} row(s) sit only on examples not mapped to an archetype "
                   "((none) in records/source-example.csv), so they inform no family")
        r["note"] = "; ".join(x for x in (why, r.get("note", "")) if x)
    write(OUTPUT / "source-list.csv", COLS, rows)
    return rows


def citation_review(rows) -> None:
    ver = {r["source-id"]: r for r in read(VERIFIED)}
    lst = {r["source-id"]: r for r in rows}
    L = []
    A = L.append
    A("# Citation review\n")
    A("Generated by the build (`src/technology_database/pipeline/sources.py`). Every citation in the "
      "database was checked against the publisher, repository or agency page. This "
      "records what changed, what could not be confirmed, and which findings need a "
      "decision rather than an edit.\n")
    def weight(sid):
        return int(lst.get(sid, {}).get("rows-in-database") or 0)

    live = [s for s in ver if lst.get(s, {}).get("role") != "superseded"]
    by = lambda st: sorted((s for s in live if ver[s]["verification-status"] == st),
                           key=lambda s: (-weight(s), s))

    L = []
    A = L.append
    A("# Citation review\n")
    A("Generated by the build (`src/technology_database/pipeline/sources.py`). Every citation in the "
      "database was checked against the publisher, repository or agency page. This "
      "records what changed, what could not be confirmed, and which findings need a "
      "decision rather than an edit.\n")
    A(f"- **{len(by('confirmed'))} confirmed** — title, authors and publisher all matched "
      "a page that was actually fetched.")
    A(f"- **{len(by('partial'))} partial** — the document was identified but at least one "
      "field could not be seen verbatim.")
    A(f"- **{len(by('unverified'))} unverified** — no reachable record. The citation is "
      "left as it was recorded, and says so.")
    A(f"- **{sum(1 for s in live if ver[s]['doi'])} carry a DOI.** A DOI is only ever "
      "what a fetched page showed. None was reconstructed from a pattern, so a blank "
      "means unconfirmed, not absent.\n")

    A("## Merges applied\n")
    A("Each pair below was confirmed to be one document and merged on 2026-08-29. "
      "The retired short form is rewritten at extraction so its rows move with it; "
      "the retired id stays registered in the manifest, labelled `superseded`.\n")
    A("| short form | id | why |")
    A("|---|---|---|")
    for sf, ids, why in MERGED:
        A(f"| {sf} | `{ids}` | {why} |")
    A("")
    A("## Renames applied\n")
    A("Same document, different name. A citation names an organization as it was at "
      "publication, and an id keys on that citation, so a later rename of the issuer does "
      "not move the id. The retired id is superseded, not deleted, and its replacement is "
      "registered in the manifest with the reason.\n")
    A("| rename | why |")
    A("|---|---|")
    for what, why in RENAMED:
        A(f"| `{what}` | {why} |")
    A("")
    A("## Where a web check lost to the document in hand\n")
    A("A reachable web page is evidence, not authority. Where a source is held locally "
      "the held copy decides, and the disagreement is recorded rather than quietly "
      "resolved.\n")
    for sid, why in STRUCTURAL:
        A(f"- **{sid}** — {why}")
    A("")

    A("## Corrections made to the recorded citation\n")
    A("The database now prints the verified form; `citation-as-cited` in "
      "`source-list.csv` keeps the string UKERC or the manifest recorded, so nothing "
      "is lost and the correction stays visible.\n")
    A("| source | rows | correction |")
    A("|---|---:|---|")
    for sid in sorted(live, key=lambda s: (-weight(s), s)):
        n = ver[sid].get("verification-note", "")
        if n and ver[sid]["verification-status"] != "unverified":
            A(f"| {lst.get(sid, {}).get('short-name', sid)} | {weight(sid)} | {n} |")
    A("")

    A("## Still unresolved\n")
    A("Grey literature that has gone offline, or that no reachable catalogue indexes. "
      "Each is left with the citation as recorded rather than a guess.\n")
    A("| source | rows | why |")
    A("|---|---:|---|")
    for sid in by("unverified"):
        A(f"| {lst.get(sid, {}).get('short-name', sid)} | {weight(sid)} | "
          f"{ver[sid].get('verification-note', '')} |")
    A("")

    (OUTPUT / "citation-review.md").write_text("\n".join(L), encoding="utf-8")
