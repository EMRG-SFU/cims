import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium", css_file="input-template.css")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path

    return Path, mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    # Technology Data Input Template

    Build a long-format technology dataset (columns: `process, technology,
    parameter, target, unit, value, value-min, value-max, formula, reference,
    note`) by entering, editing and exporting technologies one at a time.

    The parameter list mirrors `data/technology_database/vocabulary/parameter.csv`. Change that
    dictionary and the `PARAMETERS` list below together, or extractions stop
    lining up with the database they feed.

    **Workflow**

    1. **Load** an existing CSV (optional) -- must be in this notebook's export
       format, or it is rejected with a warning. Files written before the
       `target` and range columns existed still load; those columns come back
       blank.
    2. Review the **Calculation variables** (constants for formulas).
    3. **Choose a technology to edit** (optional) -- loads its values into the
       editor; calculated values come back into the **formula** field.
    4. **Edit** the entry fields. Put a literal in **value**, or an expression in
       the **formula** field (e.g. `=capacity * 0.95`) -- the formula is computed
       into `value`, and the `note` field stays free for human notes.
    5. **Save technology to working data** -- overwrites that process/technology
       if it already exists, otherwise appends it.
    6. Edit the **meta** data.
    7. **Export** the working data to CSV.

    **target** names what a row refers to when the parameter alone is not
    enough: the commodity for `flow-input` / `flow-output`, the species for
    `emissions-combustion`. Leave it blank otherwise.

    **Repeating a parameter.** A technology usually consumes more than one
    commodity. Use the custom rows: name several of them `flow-input` and give
    each a different `target`. The same works for `emissions-combustion` across
    species. Custom rows are keyed internally as `name@target`, so they no
    longer overwrite one another.

    **value-min / value-max** hold the range where the source gives one. Say
    in `note` whether the range is the source's own or an assumption.

    **Things this layer does not convert.** Units and currencies stay exactly
    as the source reports them (`2018_USD`, `psig`, `MMBtu/hr`). Record the
    cost basis of every `capex` in `note` -- equipment, installed, total-plant
    or overnight -- because an equipment cost and an installed cost can differ
    by a factor of two and look identical afterwards.
    """)
    return


@app.cell
def _():
    # ---- Configuration -----------------------------------------------------
    # The standard parameter set (and default units). Edit to change the form.
    PARAMETERS = [
        # Mirrors the parameter dictionary in data/technology_database/vocabulary/parameter.csv.
        # Change that dictionary and this list together.
        # -- identity / status ------------------------------------------
        ("description", ""),
        ("trl", ""),
        ("state-of-development", ""),
        # -- technical ---------------------------------------------------
        ("lifetime", "years"),
        ("capacity", "MW"),
        ("capacity-factor", "fraction"),
        ("annual-output", "GJ"),
        ("efficiency", "fraction"),
        ("efficiency-chp", "fraction"),
        ("heat-to-power-ratio", ""),
        ("specific-energy-consumption", "GJ/t"),
        ("heat-output-temp", "degC"),
        ("heat-output-temp-min", "degC"),
        ("heat-output-temp-max", "degC"),
        ("heat-process-temp", "degC"),
        ("heat-process-temp-min", "degC"),
        ("heat-process-temp-max", "degC"),
        ("heat-waste-temp", "degC"),
        ("heat-waste-temp-min", "degC"),
        ("heat-waste-temp-max", "degC"),
        ("pressure-output", "psig"),
        ("availability", "fraction"),
        ("load-range-min", "fraction"),
        ("degradation-rate", "fraction/yr"),
        # -- cost --------------------------------------------------------
        ("capex", "2018_USD"),
        ("opex-fixed", "2018_USD"),
        ("opex-variable", "2018_USD"),
        ("capital-recovery", "fraction/yr"),
        ("discount-rate", "fraction"),
        ("construction-time", "years"),
        ("cost-learning-rate", "fraction"),
        ("decommissioning-cost", "2018_USD"),
        # -- emissions ---------------------------------------------------
        ("emissions-combustion", "t/GJ"),
        ("emissions-process", "t/t"),
        ("emissions-total", "t/t"),
        ("emissions-captured", "t/t"),
        ("capture-rate", "fraction"),
        # -- deployment --------------------------------------------------
        ("year-available", "year"),
        ("year-unavailable", "year"),
        # -- flows (repeat as custom rows, one per commodity, with target) -
        ("flow-input", "GJ"),
        ("flow-output", "GJ"),
    ]

    STATE_OPTIONS = "existing, commercial, pilot, r-and-d, conceptual"

    # 'target' names the commodity or emission species a row refers to --
    # required for flow-input / flow-output / emissions-combustion, blank
    # otherwise.  'value-min' / 'value-max' hold a range where the source
    # gives one.
    COLUMNS = ["process", "technology", "parameter", "target", "unit",
               "value", "value-min", "value-max", "formula", "reference", "note"]
    # Columns a loaded file must contain.  Files written before 'target' and
    # the range columns existed still load; the missing columns come back blank.
    REQUIRED_COLUMNS = ["process", "technology", "parameter", "unit", "value",
                        "reference", "note"]
    return COLUMNS, PARAMETERS, REQUIRED_COLUMNS, STATE_OPTIONS


@app.cell
def _(Path, mo):
    # Folder files are read from / written into: the notebook's own folder
    # (mo.notebook_dir() is robust even when __file__ is not set).
    DATA_DIR = mo.notebook_dir() or Path.cwd()
    return (DATA_DIR,)


@app.cell
def _():
    # ---- Calculation variables --------------------------------------------
    # Shared constants available to '=' formulas in any value field.
    # The value of each parameter field is ALSO available to other fields'
    # formulas (e.g. 'capacity', 'capacity_factor').
    calc_vars = {
        "hours_per_year": 8760,
        "GJ_per_MWh": 3.6,
        "GJ_per_MMBtu": 1.0551,
    }
    return (calc_vars,)


@app.cell
def _():
    # ---- Field-aware formula resolver -------------------------------------
    # '=expr' is evaluated against calc_vars, the math module, the value of any
    # other field (slug: spaces -> '_'), and a few safe builtins. Literals
    # coerce to int/float. Iterates so a formula may depend on another formula.
    import math as _math

    def _coerce(s):
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s

    def resolve_fields(raw_by_name, calc_vars):
        """Return {name: (value, formula_or_None, error_or_None)}.

        The returned formula is the expression WITHOUT the leading '=' (that
        is the form stored in the 'formula' column)."""
        ns = {k: getattr(_math, k) for k in dir(_math) if not k.startswith("_")}
        ns.update({"round": round, "abs": abs, "min": min, "max": max, "sum": sum})
        ns.update(calc_vars)
        slug = {name: name.replace(" ", "_").replace("-", "_")
                for name in raw_by_name}

        results = {}
        pending = {}
        for name, raw in raw_by_name.items():
            s = (raw or "").strip()
            if s == "":
                results[name] = (None, None, None)
            elif s.startswith("="):
                pending[name] = s
            else:
                val = _coerce(s)
                results[name] = (val, None, None)
                ns[slug[name]] = val

        progress = True
        while pending and progress:
            progress = False
            for name in list(pending):
                expr = pending[name][1:].strip()
                try:
                    val = eval(expr, {"__builtins__": {}}, ns)  # noqa: S307
                except NameError:
                    continue
                except Exception as e:  # noqa: BLE001
                    results[name] = (None, expr, f"{type(e).__name__}: {e}")
                    del pending[name]
                    progress = True
                    continue
                results[name] = (val, expr, None)
                ns[slug[name]] = val
                del pending[name]
                progress = True

        for name, s in pending.items():
            expr = s[1:].strip()
            try:
                results[name] = (eval(expr, {"__builtins__": {}}, ns), expr, None)  # noqa: S307
            except Exception as e:  # noqa: BLE001
                results[name] = (None, expr, f"{type(e).__name__}: {e}")

        return results

    return (resolve_fields,)


@app.cell
def _():
    # ---- Editor / data helpers --------------------------------------------
    def field_raw(field):
        """Resolver input for a parameter: the formula (as '=expr') when the
        formula field is filled, otherwise the literal 'value' field."""
        f = (field.get("formula") or "").strip()
        if f:
            return f if f.startswith("=") else "=" + f
        return field.get("value") or ""

    def parse_note(note):
        """Back-compat: split a legacy 'calc: =expr | human note' string into
        (formula, human_note). Used when loading older files that stored the
        formula inside the note column."""
        if not note:
            return None, ""
        formula = None
        rest = []
        for part in str(note).split(" | "):
            if part.startswith("calc: "):
                formula = part[len("calc: "):]
            else:
                rest.append(part)
        return formula, " | ".join(rest)

    def gather_entries(param_inputs, custom_inputs, params):
        """All (key, fields) rows to process: the standard parameters plus any
        named custom rows (custom rows with a blank name are skipped).

        The key is unique within the form -- it is what the formula resolver
        indexes by.  The parameter name itself is fields['name'], so the same
        parameter may appear several times with different targets (one
        flow-input row per commodity, one emissions-combustion row per species)."""
        entries = []
        for nm, _ in params:
            _f = dict(param_inputs[nm].value)
            _f["name"] = nm
            entries.append((nm, _f))
        used = {nm for nm, _ in params}
        for _ck in custom_inputs.value:
            _f = dict(custom_inputs[_ck].value)
            _cn = (_f.get("name") or "").strip()
            if not _cn:
                continue
            _tg = (_f.get("target") or "").strip()
            key = _cn if _cn not in used else (f"{_cn}@{_tg}" if _tg else f"{_cn}@{_ck}")
            while key in used:
                key += "_"
            used.add(key)
            _f["name"] = _cn
            entries.append((key, _f))
        return entries

    def read_meta(csv_path, pd, Path):
        """Read the '<stem>_meta.csv' sidecar next to a loaded CSV.
        Returns {parameter: value}; empty dict if absent or unreadable."""
        _p = Path(csv_path)
        _mp = _p.with_name(_p.stem + "_meta.csv")
        if not _mp.exists():
            return {}
        try:
            _m = pd.read_csv(_mp)
        except Exception:  # noqa: BLE001
            return {}
        if "parameter" not in _m.columns or "value" not in _m.columns:
            return {}
        _out = {}
        for _k, _v in zip(_m["parameter"], _m["value"]):
            if pd.isna(_k):
                continue
            _out[str(_k).strip()] = "" if pd.isna(_v) else str(_v)
        return _out

    return field_raw, gather_entries, parse_note, read_meta


@app.cell
def _(PARAMETERS, calc_vars, mo, pd):
    # ---- (Step 2) Calculation-variables panel -----------------------------
    _consts = pd.DataFrame(list(calc_vars.items()), columns=["name", "value"])
    _field_vars = ", ".join(
        name.replace(" ", "_").replace("-", "_") for name, _ in PARAMETERS)
    calc_view = mo.vstack([
        mo.md("## Calculation variables"),
        mo.md("**Constants** available to every `=` formula:"),
        mo.ui.table(_consts, selection=None, pagination=False),
        mo.md(
            "Each parameter's value is **also** available to formulas in other "
            "fields, referenced by name with spaces and hyphens replaced by "
            "underscores:\n\n"
            f"`{_field_vars}`\n\n"
            "Example: for the annual output formula cell value, use "
            "**= capacity * capacity_factor * hours_per_year * GJ_per_MWh**"
        ),
    ])
    return (calc_view,)


@app.cell
def _(mo):
    # ---- State: the working dataset (long-format row dicts) ---------------
    get_data, set_data = mo.state([])
    return get_data, set_data


@app.cell
def _(mo):
    # ---- State: the "editor seed" -- the technology snapshot loaded into the
    # entry fields. Kept SEPARATE from the working data so that saving never
    # rebuilds the form (which is what caused earlier reactive loops/freezes).
    get_seed, set_seed = mo.state(None)
    return get_seed, set_seed


@app.cell
def _(mo):
    # ---- State: name of the currently loaded CSV (persisted for display) ---
    get_loaded, set_loaded = mo.state(None)
    return get_loaded, set_loaded


@app.cell
def _(mo):
    # ---- State: meta (licence / citation) read from the '<name>_meta.csv'
    # sidecar when a CSV is loaded, so re-editing does not drop the citation.
    get_meta, set_meta = mo.state({})
    return get_meta, set_meta


@app.cell
def _():
    # ---- Native OS file dialog (cross-platform) ---------------------------
    # macOS uses `osascript` (no tkinter; runs in its own process, so no
    # main-thread crash); Windows & Linux use tkinter, with a zenity fallback
    # on Linux. Returns the chosen path (str), "" if cancelled, or None if no
    # dialog is available (e.g. a headless/remote session).
    def native_path(mode, *, title="Select", initialdir="",
                    initialfile="", filetypes=None, defaultextension=".csv"):
        import sys

        if sys.platform == "darwin":
            return _osascript(mode, title, initialdir, initialfile)
        _res = _tk(mode, title, initialdir, initialfile, filetypes, defaultextension)
        if _res is not None:
            return _res
        if sys.platform.startswith("linux"):
            return _zenity(mode, title, initialdir, initialfile)
        return None

    def _osascript(mode, title, initialdir, initialfile):
        import os
        import subprocess

        def q(text):
            return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'

        loc = ""
        if initialdir and os.path.isdir(initialdir):
            loc = " default location (POSIX file %s)" % q(initialdir)
        if mode == "open":
            script = "POSIX path of (choose file with prompt %s%s)" % (q(title), loc)
        elif mode == "dir":
            script = "POSIX path of (choose folder with prompt %s%s)" % (q(title), loc)
        else:
            nm = " default name %s" % q(initialfile) if initialfile else ""
            script = ("POSIX path of (choose file name with prompt %s%s%s)"
                      % (q(title), nm, loc))
        try:
            r = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True)
        except Exception:
            return None
        if r.returncode == 0:
            return r.stdout.strip()
        if "cancel" in (r.stderr or "").lower():
            return ""
        return None

    def _tk(mode, title, initialdir, initialfile, filetypes, defaultextension):
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception:
            return None
        try:
            _root = tk.Tk()
            _root.withdraw()
            try:
                _root.attributes("-topmost", True)
            except Exception:
                pass
            if mode == "open":
                _kw = {"title": title}
                if initialdir:
                    _kw["initialdir"] = initialdir
                if filetypes:
                    _kw["filetypes"] = filetypes
                _p = filedialog.askopenfilename(**_kw)
            elif mode == "dir":
                _p = filedialog.askdirectory(title=title, initialdir=initialdir or None)
            else:
                _kw = {"title": title}
                if initialdir:
                    _kw["initialdir"] = initialdir
                if initialfile:
                    _kw["initialfile"] = initialfile
                if filetypes:
                    _kw["filetypes"] = filetypes
                if defaultextension:
                    _kw["defaultextension"] = defaultextension
                _p = filedialog.asksaveasfilename(**_kw)
            _root.update()
            _root.destroy()
            return _p or ""
        except Exception:
            return None

    def _zenity(mode, title, initialdir, initialfile):
        import os
        import shutil
        import subprocess

        exe = shutil.which("zenity")
        if not exe:
            return None
        args = [exe, "--file-selection", "--title", title]
        if mode == "dir":
            args.append("--directory")
        if mode == "save":
            args += ["--save", "--confirm-overwrite"]
        _start = os.path.join(initialdir or "", initialfile or "")
        if _start:
            args += ["--filename", _start]
        try:
            r = subprocess.run(args, capture_output=True, text=True)
        except Exception:
            return None
        if r.returncode == 0:
            return r.stdout.strip()
        return ""

    return (native_path,)


@app.cell
def _(mo):
    # ---- (Step 1) Load control: opens the OS file picker ------------------
    load_button = mo.ui.run_button(label="Load CSV")
    return (load_button,)


@app.cell
def _(
    COLUMNS,
    DATA_DIR,
    Path,
    REQUIRED_COLUMNS,
    load_button,
    mo,
    native_path,
    parse_note,
    pd,
    read_meta,
    set_data,
    set_loaded,
    set_meta,
):
    # ---- (Step 1) Load logic: accepts our export format -------------------
    # Requires the core columns; 'formula' is optional. Older files that stored
    # a formula inside the note ('calc: =...') are migrated into the formula col.
    load_status = mo.md("")
    _path = None
    if load_button.value:
        _path = native_path(
            "open", title="Select a CSV to load", initialdir=str(DATA_DIR),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if _path is None:
            load_status = mo.md(
                "**Warning:** the OS file dialog is unavailable in this "
                "environment (no desktop session)."
            )
        elif _path == "":
            load_status = mo.md("*Load cancelled.*")
    if _path:
        try:
            _df = pd.read_csv(_path)
        except Exception as e:  # noqa: BLE001
            load_status = mo.md(f"**Warning:** could not read file: {e}")
        else:
            _missing = [c for c in REQUIRED_COLUMNS if c not in _df.columns]
            if _missing:
                load_status = mo.md(
                    "**Warning: wrong format -- nothing loaded.**\n\n"
                    f"Missing required columns: `{_missing}`\n\n"
                    f"Found: `{list(_df.columns)}`"
                )
            else:
                if "formula" not in _df.columns:
                    _df["formula"] = None
                _df = _df.where(pd.notna(_df), None)
                for _c in COLUMNS:
                    if _c not in _df.columns:
                        _df[_c] = None
                _rows = _df[COLUMNS].to_dict("records")
                for _r in _rows:  # migrate legacy 'calc:' notes -> formula column
                    if not _r.get("formula") and _r.get("note"):
                        _f, _n = parse_note(_r.get("note"))
                        if _f:
                            _r["formula"] = _f
                            _r["note"] = _n or None
                set_data(_rows)
                set_loaded(str(_path))
                _meta = read_meta(_path, pd, Path)
                set_meta(_meta)
                _meta_note = (
                    " Meta (licence / citation) loaded from the sidecar."
                    if _meta else
                    " *No `_meta.csv` sidecar found -- licence / citation left "
                    "at their defaults.*"
                )
                load_status = mo.md(
                    f"Loaded **{len(_rows)} rows** from `{_path}`."
                    + _meta_note
                )
    return (load_status,)


@app.cell
def _(get_loaded, mo):
    # Persistent, compact reminder of which file is currently loaded.
    # Rendered as a small badge styled via the .loaded-badge CSS class
    # (see input-template.css) so it stays small.
    import html as _html

    _ln = get_loaded()
    if _ln:
        loaded_view = mo.Html(
            '<div class="loaded-badge loaded-yes">Loaded: '
            f'<code>{_html.escape(str(_ln))}</code></div>'
        )
    else:
        loaded_view = mo.Html(
            '<div class="loaded-badge loaded-no">No file loaded</div>'
        )
    return (loaded_view,)


@app.cell
def _(get_data, mo):
    # ---- (Step 3) Technology chooser --------------------------------------
    _pairs = []
    _seen = set()
    for _r in get_data():
        _key = (_r.get("process"), _r.get("technology"))
        if _r.get("technology") and _key not in _seen:
            _seen.add(_key)
            _pairs.append(_key)
    _opts = {f"{_p} / {_t}": (_p, _t) for _p, _t in _pairs}
    tech_picker = mo.ui.dropdown(options=_opts, value=None, label="Technology to edit")
    edit_button = mo.ui.run_button(label="Load into editor")
    new_button = mo.ui.run_button(label="Clear editor (new technology)")
    return edit_button, new_button, tech_picker


@app.cell
def _(
    PARAMETERS,
    edit_button,
    get_data,
    new_button,
    parse_note,
    set_seed,
    tech_picker,
):
    # ---- (Step 3) Seed logic: build the editor snapshot for the chosen tech.
    # Writes only set_seed (never set_data), so it cannot loop with saving.
    if new_button.value:
        set_seed(None)
    elif edit_button.value and tech_picker.value:
        _proc, _tech = tech_picker.value
        _rows_t = [
            r for r in get_data()
            if r.get("process") == _proc and r.get("technology") == _tech
        ]
        _std_names = {nm for nm, _ in PARAMETERS}
        # First row for a standard parameter fills that slot; every other row
        # -- an unknown parameter, or a repeat with a different target --
        # becomes a custom row, so nothing is silently dropped on reload.
        _by = {}
        _extra = []
        for _r in _rows_t:
            _pn = _r.get("parameter")
            if _pn in _std_names and _pn not in _by:
                _by[_pn] = _r
            else:
                _extra.append(_r)

        def _row_fields(_r):
            _formula = _r.get("formula")
            _note = _r.get("note")
            if not _formula and _note:  # back-compat: formula stashed in note
                _formula, _note = parse_note(_note)
            _has_f = bool(_formula)
            return {
                "target": "" if _r.get("target") is None else str(_r.get("target")),
                "value-min": "" if _r.get("value-min") is None else str(_r.get("value-min")),
                "value-max": "" if _r.get("value-max") is None else str(_r.get("value-max")),
                "value": "" if _has_f else (
                    "" if _r.get("value") is None else str(_r.get("value"))
                ),
                "formula": str(_formula) if _has_f else "",
                "unit": "" if _r.get("unit") is None else str(_r.get("unit")),
                "reference": "" if _r.get("reference") is None else str(_r.get("reference")),
                "note": "" if not _note else str(_note),
            }

        _params = {}
        for _name, _ in PARAMETERS:
            _r = _by.get(_name)
            _params[_name] = (
                {"target": "", "value": "", "value-min": "", "value-max": "",
                 "formula": "", "unit": "", "reference": "", "note": ""}
                if _r is None else _row_fields(_r)
            )

        _custom = []
        for _r in _extra:
            _cf = _row_fields(_r)
            _cf["name"] = str(_r.get("parameter"))
            _custom.append(_cf)

        set_seed({"process": _proc, "technology": _tech,
                  "params": _params, "custom": _custom[:12]})   # matches N_CUSTOM
    return


@app.cell
def _(PARAMETERS, get_seed, mo):
    # ---- (Step 4) Entry fields, built from the editor seed ----------------
    # Depends ONLY on the seed (not the working data), so saving the working
    # data never rebuilds / clears these fields.
    _seed = get_seed() or {}
    _sp = _seed.get("params", {})
    process_in = mo.ui.text(
        value=_seed.get("process", "") or "",
        label="process", placeholder="e.g. steam",
    )
    technology_in = mo.ui.text(
        value=_seed.get("technology", "") or "",
        label="technology", placeholder="e.g. electricity_boiler",
    )
    param_inputs = mo.ui.dictionary({
        name: mo.ui.dictionary({
            "target": mo.ui.text(
                value=_sp.get(name, {}).get("target", ""),
                placeholder="commodity / species", label="",
            ),
            "value": mo.ui.text(
                value=_sp.get(name, {}).get("value", ""),
                placeholder="value", label="",
            ),
            "formula": mo.ui.text(
                value=_sp.get(name, {}).get("formula", ""),
                placeholder="formula", label="",
            ),
            "value-min": mo.ui.text(
                value=_sp.get(name, {}).get("value-min", ""),
                placeholder="min", label="",
            ),
            "value-max": mo.ui.text(
                value=_sp.get(name, {}).get("value-max", ""),
                placeholder="max", label="",
            ),
            "unit": mo.ui.text(
                value=_sp.get(name, {}).get("unit", default_unit) or default_unit,
                label="",
            ),
            "reference": mo.ui.text(
                value=_sp.get(name, {}).get("reference", ""),
                placeholder="e.g. page 43", label="",
            ),
            "note": mo.ui.text(
                value=_sp.get(name, {}).get("note", ""),
                placeholder="note / assumption", label="",
            ),
        })
        for name, default_unit in PARAMETERS
    })
    return param_inputs, process_in, technology_in


@app.cell
def _(get_seed, mo):
    # ---- (Step 4) Custom parameter rows (name them yourself) --------------
    # Five extra rows for parameters not in the standard list. A row is only
    # saved if you give it a name. Built from the seed so editing a loaded
    # technology repopulates its custom parameters too.
    N_CUSTOM = 12
    _cust = (get_seed() or {}).get("custom", [])

    def _cv(_i, _key, _default=""):
        return _cust[_i].get(_key, _default) if _i < len(_cust) else _default

    custom_inputs = mo.ui.dictionary({
        f"custom_{_i}": mo.ui.dictionary({
            "name": mo.ui.text(value=_cv(_i, "name"),
                               placeholder="custom parameter name", label=""),
            "target": mo.ui.text(value=_cv(_i, "target"),
                                 placeholder="commodity / species", label=""),
            "value": mo.ui.text(value=_cv(_i, "value"),
                                placeholder="value", label=""),
            "formula": mo.ui.text(value=_cv(_i, "formula"),
                                  placeholder="formula", label=""),
            "value-min": mo.ui.text(value=_cv(_i, "value-min"),
                                    placeholder="min", label=""),
            "value-max": mo.ui.text(value=_cv(_i, "value-max"),
                                    placeholder="max", label=""),
            "unit": mo.ui.text(value=_cv(_i, "unit"),
                               placeholder="unit", label=""),
            "reference": mo.ui.text(value=_cv(_i, "reference"),
                                    placeholder="e.g. page 43", label=""),
            "note": mo.ui.text(value=_cv(_i, "note"),
                               placeholder="note / assumption", label=""),
        })
        for _i in range(N_CUSTOM)
    })
    return (custom_inputs,)


@app.cell
def _(PARAMETERS, STATE_OPTIONS, custom_inputs, mo, param_inputs):
    # ---- (Step 4) Lay out the parameter grid ------------------------------
    import html as _html

    # Per-parameter help shown as a hover tooltip on the name -- handy when a
    # long placeholder is truncated inside the narrow input box. Extend freely.
    _HELP = {
        "state-of-development": "options: " + STATE_OPTIONS,
        "flow-input": "one row per commodity consumed; put the commodity in target",
        "flow-output": "one row per commodity produced; put the commodity in target",
        "emissions-combustion": "put the species in target: co2, co2e or pfc. Scope 1 GHGs only - non-GHG species (nox, so2, pm) are out of scope and are not converted to CO2e",
        "capex": "record the cost basis in note: equipment / installed / total-plant / overnight",
    }

    _W = [2, 2, 2, 1, 1, 1, 2, 2, 3]

    _header = mo.hstack(
        [mo.md("**parameter**"), mo.md("**target**"), mo.md("**value**"),
         mo.md("**min**"), mo.md("**max**"), mo.md("**unit**"),
         mo.md("**formula**"), mo.md("**reference**"), mo.md("**note**")],
        widths=_W, gap=0.5,
    )
    _rows = [_header]
    for _name, _ in PARAMETERS:
        _d = param_inputs[_name]
        if _name in _HELP:
            _label = mo.Html(
                f'<span title="{_html.escape(_HELP[_name], quote=True)}">'
                f'{_html.escape(_name)} (i)</span>'
            )
        else:
            _label = mo.md(_name)
        _rows.append(
            mo.hstack(
                [_label, _d["target"], _d["value"], _d["value-min"],
                 _d["value-max"], _d["unit"], _d["formula"],
                 _d["reference"], _d["note"]],
                widths=_W, gap=0.5,
            )
        )
    _rows.append(mo.md("*custom parameters (type a name to include the row):*"))
    for _key in custom_inputs.value:
        _c = custom_inputs[_key]
        _rows.append(
            mo.hstack(
                [_c["name"], _c["target"], _c["value"], _c["value-min"],
                 _c["value-max"], _c["unit"], _c["formula"],
                 _c["reference"], _c["note"]],
                widths=_W, gap=0.5,
            )
        )
    param_grid = mo.vstack(_rows, gap=0.25)
    return (param_grid,)


@app.cell
def _(
    PARAMETERS,
    calc_vars,
    custom_inputs,
    field_raw,
    gather_entries,
    mo,
    param_inputs,
    pd,
    resolve_fields,
):
    # ---- (Step 4) Live resolved-values preview ----------------------------
    _entries = gather_entries(param_inputs, custom_inputs, PARAMETERS)
    _raw = {_nm: field_raw(_f) for _nm, _f in _entries}
    _res = resolve_fields(_raw, calc_vars)
    _prev = []
    for _nm, _f in _entries:
        if (_raw[_nm] or "").strip() == "":
            continue
        _val, _formula, _err = _res[_nm]
        _prev.append({
            "parameter": _nm,
            "input": _raw[_nm],
            "resolved": _err if _err else _val,
            "status": "error" if _err else ("formula" if _formula else "literal"),
        })
    preview = mo.vstack([
        mo.md("### Resolved values"),
        mo.ui.table(pd.DataFrame(_prev), selection=None, page_size=25) if _prev
        else mo.md("*Enter some values above to preview them.*"),
    ])
    return (preview,)


@app.cell
def _(mo):
    # ---- (Step 5) Save / clear controls -----------------------------------
    save_button = mo.ui.run_button(label="Save technology to working data")
    clear_data_button = mo.ui.run_button(label="Clear ALL working data")
    return clear_data_button, save_button


@app.cell
def _(
    PARAMETERS,
    calc_vars,
    custom_inputs,
    field_raw,
    gather_entries,
    get_data,
    mo,
    param_inputs,
    process_in,
    resolve_fields,
    save_button,
    set_data,
    technology_in,
):
    # ---- (Step 5) Save logic: overwrite this process/technology or append --
    # Reads the form (seed-based) + get_data() (self-loop protected) and writes
    # set_data. set_data does NOT rebuild the form, so there is no loop.
    save_status = mo.md("")
    if save_button.value:
        _proc = (process_in.value or "").strip()
        _tech = (technology_in.value or "").strip()
        if not _tech:
            save_status = mo.md("**Enter a technology id before saving.**")
        else:
            _entries = gather_entries(param_inputs, custom_inputs, PARAMETERS)
            _raw = {_nm: field_raw(_f) for _nm, _f in _entries}
            _res = resolve_fields(_raw, calc_vars)
            _new = []
            for _nm, _f in _entries:
                _val, _formula, _err = _res[_nm]
                if _val is None and _err is None:
                    continue
                if _err:
                    _val = f"ERROR: {_err}"
                _new.append({
                    "process": _proc,
                    "technology": _tech,
                    "parameter": _f.get("name") or _nm,
                    "target": (_f.get("target") or "").strip() or None,
                    "unit": (_f.get("unit") or "").strip() or None,
                    "value": _val,
                    "value-min": (_f.get("value-min") or "").strip() or None,
                    "value-max": (_f.get("value-max") or "").strip() or None,
                    "formula": _formula,  # 'expr' (no leading '=') or None
                    "reference": (_f.get("reference") or "").strip() or None,
                    "note": (_f.get("note") or "").strip() or None,
                })
            _existing = [
                r for r in get_data()
                if r.get("process") == _proc and r.get("technology") == _tech
            ]
            _kept = [
                r for r in get_data()
                if not (r.get("process") == _proc and r.get("technology") == _tech)
            ]
            set_data(_kept + _new)
            _action = "Overwrote" if _existing else "Added"
            save_status = mo.md(f"{_action} **{_proc} / {_tech}** ({len(_new)} rows).")
    return (save_status,)


@app.cell
def _(clear_data_button, set_data):
    if clear_data_button.value:
        set_data([])
    return


@app.cell
def _(COLUMNS, get_data, mo, pd):
    # ---- Working-data view (read-only) ------------------------------------
    _df = pd.DataFrame(get_data(), columns=COLUMNS)
    working_view = mo.vstack([
        mo.md("## Working data"),
        mo.md(f"**rows:** {len(_df)}  |  "
              f"**technologies:** {_df['technology'].nunique() if len(_df) else 0}"),
        mo.ui.table(_df, selection=None, pagination=True, page_size=25) if len(_df)
        else mo.md("*Empty -- load a file or save a technology.*"),
    ])
    return (working_view,)


@app.cell
def _(DATA_DIR, Path, get_loaded, get_meta, mo):
    # ---- (Step 6) Meta fields ---------------------------------------------
    # Seeded from the loaded file's '_meta.csv' sidecar so the citation
    # survives a load -> edit -> export round trip.
    _meta_seed = get_meta() or {}
    licence_in = mo.ui.text(
        value=_meta_seed.get("licence", "US Government"),
        label="licence", full_width=True,
    )
    citation_in = mo.ui.text(
        value=_meta_seed.get("citation", ""),
        label="citation", full_width=True,
        placeholder="Author(s). (Year). Title. DOI/URL",
    )
    # Output path defaults to the loaded file (folder + name) so export
    # overwrites it; both fields accept any path on disk.
    _lp = Path(get_loaded()) if get_loaded() else None
    out_dir_in = mo.ui.text(
        value=str(_lp.parent) if _lp else str(DATA_DIR),
        label="output folder", full_width=True,
    )
    out_name_in = mo.ui.text(
        value=_lp.name if _lp else "technology_data.csv",
        label="output filename", full_width=True,
    )
    return citation_in, licence_in, out_dir_in, out_name_in


@app.cell
def _(mo):
    export_button = mo.ui.run_button(label="Export CSV")
    save_as_button = mo.ui.run_button(label="Save As")
    return export_button, save_as_button


@app.cell
def _(
    COLUMNS,
    DATA_DIR,
    Path,
    citation_in,
    export_button,
    get_data,
    get_loaded,
    licence_in,
    mo,
    native_path,
    out_dir_in,
    out_name_in,
    pd,
    save_as_button,
):
    # ---- (Step 7) Export -- Save As... OR type a path + Export CSV ---
    export_status = mo.md("")
    if export_button.value or save_as_button.value:
        _rows = get_data()
        _df = pd.DataFrame(_rows, columns=COLUMNS)
        if len(_df) == 0:
            export_status = mo.md("Nothing to export -- working data is empty.")
        else:
            _target = None
            _msg = None
            if save_as_button.value:
                _default_name = (out_name_in.value or "").strip() or (
                    Path(get_loaded()).name if get_loaded() else "technology_data.csv"
                )
                _init_dir = (out_dir_in.value or "").strip() or (
                    str(Path(get_loaded()).parent) if get_loaded() else str(DATA_DIR)
                )
                _sp = native_path(
                    "save", title="Save CSV as", defaultextension=".csv",
                    initialfile=_default_name, initialdir=_init_dir,
                    filetypes=[("CSV files", "*.csv")],
                )
                if _sp is None:
                    _msg = mo.md(
                        "**Warning:** the OS save dialog is unavailable here; "
                        "use the folder/filename fields + Export CSV."
                    )
                elif _sp == "":
                    _msg = mo.md("*Save cancelled.*")
                else:
                    _target = Path(_sp)
            else:
                _name = (out_name_in.value or "technology_data.csv").strip()
                if not _name.lower().endswith(".csv"):
                    _name += ".csv"
                _np = Path(_name).expanduser()
                _target = _np if _np.is_absolute() else Path(
                    (out_dir_in.value or ".").strip()
                ).expanduser() / _name
            if _msg is not None:
                export_status = _msg
            elif _target is not None:
                if _target.suffix.lower() != ".csv":
                    _target = _target.with_suffix(".csv")
                try:
                    _target.parent.mkdir(parents=True, exist_ok=True)
                    _df.to_csv(_target, index=False, columns=COLUMNS)
                    _meta_path = _target.with_name(_target.stem + "_meta.csv")
                    pd.DataFrame(
                        [["licence", licence_in.value], ["citation", citation_in.value]],
                        columns=["parameter", "value"],
                    ).to_csv(_meta_path, index=False)
                    export_status = mo.md(
                        f"Wrote **{len(_df)} rows** to `{_target}` "
                        f"and meta to `{_meta_path.name}`."
                    )
                except Exception as e:  # noqa: BLE001
                    export_status = mo.md(f"**Warning:** could not write: {e}")
    return (export_status,)


@app.cell
def _(
    calc_view,
    citation_in,
    clear_data_button,
    edit_button,
    export_button,
    export_status,
    licence_in,
    load_button,
    load_status,
    loaded_view,
    mo,
    new_button,
    out_dir_in,
    out_name_in,
    param_grid,
    preview,
    process_in,
    save_as_button,
    save_button,
    save_status,
    tech_picker,
    technology_in,
    working_view,
):
    # ---- Final page layout ------------------------------------------------
    _load = mo.vstack([
        mo.md("## 1. Load (optional)"),
        load_button,
        load_status,
        loaded_view,
    ])

    _choose = mo.vstack([
        mo.md("## 3. Choose a technology to edit (optional)"),
        mo.hstack([tech_picker, edit_button, new_button], justify="start", gap=1),
    ])

    _edit = mo.vstack([
        mo.md("## 4. Edit"),
        mo.hstack([process_in, technology_in], justify="start", gap=1),
        param_grid,
        preview,
    ])

    _save = mo.vstack([
        mo.md("## 5. Save"),
        mo.hstack([save_button, clear_data_button], justify="start", gap=1),
        save_status,
    ])

    _meta = mo.vstack([
        mo.md("## 6. Meta"),
        licence_in,
        citation_in,
    ])

    _export = mo.vstack([
        mo.md("## 7. Export"),
        out_dir_in,
        out_name_in,
        mo.hstack([export_button, save_as_button], justify="start", gap=1),
        export_status,
    ])

    mo.vstack([
        _load,
        mo.md("---"),
        calc_view,
        mo.md("---"),
        _choose,
        mo.md("---"),
        _edit,
        mo.md("---"),
        _save,
        working_view,
        mo.md("---"),
        _meta,
        mo.md("---"),
        _export,
    ])
    return


if __name__ == "__main__":
    app.run()
