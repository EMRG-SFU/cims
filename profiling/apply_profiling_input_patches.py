"""Profiling-only input patches for the cloud copy of the CIMS inputs (not the user's files).

1. EMH TF2 fix: move market_share_class names from Context to Value in the
   reference clean electricity, LDV ZEV, and renewable fuel content policies.
2. Give each tech with dic_initial but no dic_class its own singleton DIC class,
   which reproduces the pre-#511 behaviour (DIC follows the tech's own NMS).
"""
import csv, glob, io, sys
root = sys.argv[1]
POL = f"{root}/data/model_inputs/policies/reference"
FIX = ["ref_clean_electricity", "ref_ldv_zev_federal", "ref_ldv_zev_prov",
       "ref_renewable_fuel_content_fed", "ref_renewable_fuel_content_prov"]

def rw(path, fn):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    header = list(rows[0].keys()) if rows else None
    n = fn(rows)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)
    return n

def fix_class(rows):
    n = 0
    for r in rows:
        if r["Parameter"] == "market_share_class" and not r["Value"] and r["Context"]:
            r["Value"], r["Context"] = r["Context"], ""; n += 1
    return n

for name in FIX:
    for p in sorted(glob.glob(f"{POL}/{name}/*.csv") + glob.glob(f"{POL}/{name}.csv")):
        print(p.split("/")[-1], rw(p, fix_class), "class rows moved")

def add_dic_class(rows):
    has = {(r["Branch"], r["Technology"]) for r in rows if r["Parameter"] == "dic_class"}
    need = []
    for r in rows:
        k = (r["Branch"], r["Technology"])
        if r["Parameter"] == "dic_initial" and k not in has:
            has.add(k)
            new = {c: "" for c in r}
            for c in ("Branch", "Type", "Region", "Sector", "Service", "Technology"):
                new[c] = r[c]
            new["Parameter"] = "dic_class"; new["Value"] = f"{r['Branch']}|{r['Technology']}"
            need.append(new)
    rows.extend(need)
    return len(need)

for p in sorted(glob.glob(f"{root}/data/model_inputs/model/DIC/*.csv")):
    print(p.split("/")[-1], rw(p, add_dic_class), "singleton dic_class rows added")
