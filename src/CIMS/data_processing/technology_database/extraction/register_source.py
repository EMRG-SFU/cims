"""Register one extracted source in the tech_data manifest, citation and priority
tables, and add its loader line. Idempotent: existing rows for the source-id are
replaced.

    python src/technology_database/extraction/register_source.py spec.json      (run from anywhere)

spec.json: {"source_id", "folder" (the publisher folder, the same under data/raw_data/technology_database/ and primary-sources/), "stem", "doc_ext",
            "original_name", "author", "institution", "title", "description",
            "year", "licence", "citation" (full), "doi", "url",
            "short_name", "breadth", "rank", "priority_note", "region",
            "verified_authors", "verified_publisher", "sidecars": {suffix: description}}
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RAW, RECORDS  # noqa: E402

REF = RECORDS / "reference"
EXTRACT = Path(__file__).resolve().parents[1] / "pipeline" / "extract.py"
TODAY = "2026-09-29"


def rw(path, key, new_rows, sort):
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    h, body = rows[0], rows[1:]
    body = [r for r in body if not key(r)] + new_rows
    for r in new_rows:
        assert len(r) == len(h), (path, len(r), len(h), r)
    if sort:
        body.sort(key=sort)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f, lineterminator="\n").writerows([h] + body)
    return h


def main(spec_path):
    s = json.load(open(spec_path, encoding="utf-8"))
    sid, folder, stem = s["source_id"], s["folder"], s["stem"]
    fp = folder   # the same folder under the raw data and under primary-sources/
    rw(RAW / "source-details_sources.csv", lambda r: r[0] == sid,
       [[sid, s["author"], s["institution"], s["title"], s["description"], s["year"],
         s["licence"], "", ""]], lambda r: r[0])
    files = [[sid, stem, s["title"], "source document", s["doc_ext"], f"{stem}.csv", fp,
              s.get("original_name", ""), f"{stem}.{s['doc_ext']}", s.get("url", ""),
              s.get("doi", ""), "", TODAY, "held; renamed from the download name " + TODAY]]
    std = {"": "extraction - technology parameters (ETSAP-brief shape, with _nodes)",
           "_meta": "extraction metadata - licence, citation and extraction notes",
           "_nodes": "node map - sector, end use and service unit per technology"}
    std.update(s.get("sidecars", {}))
    for suf, d in std.items():
        f = RAW / folder / f"{stem}{suf}.csv"
        if not f.exists():
            continue
        files.append([sid, stem + suf, s["title"] + (" - " + suf[1:] if suf else " - extraction"),
                      d, "csv", "", fp, "", f.name, "", "", "", TODAY,
                      "extraction produced " + TODAY + " from the held document"])
    for extra in s.get("other_files", []):
        files.append([sid, Path(extra).stem, s["title"] + " - " + extra.rsplit(".", 1)[-1],
                      "source data file", extra.rsplit(".", 1)[-1], "", fp, "", extra,
                      s.get("url", ""), "", "", TODAY, "held"])
    rw(RAW / "source-details_files.csv", lambda r: r[0] == sid, files, lambda r: (r[0], r[1]))
    rw(REF / "source-citation.csv", lambda r: r[0] == sid,
       [[sid, "confirmed", s["verified_authors"], s["title"], s["verified_publisher"], s["year"],
         s.get("doi", ""), s.get("url", ""), "confirmed against the held document " + TODAY]],
       lambda r: r[0])
    rw(REF / "source-priority.csv", lambda r: r[0] == sid,
       [[sid, s["short_name"], s["breadth"], s["year"], str(s["rank"]), s["priority_note"]]], None)
    txt = EXTRACT.read_text(encoding="utf-8")
    line = f'    ("{folder}/{stem}", "{sid}", "{s["region"]}"),\n'
    if line not in txt:
        txt = re.sub(r'(BRIEF_SHAPE = \[\n(?:.*\n)*?)(\])', lambda m: m.group(1) + line + m.group(2), txt, count=1)
        EXTRACT.write_text(txt, encoding="utf-8")
    print("registered", sid, "with", len(files), "file rows")


if __name__ == "__main__":
    main(sys.argv[1])
