# Conventions

The rules the data processing pipeline follows. The pipeline turns sources into the
model's input files, including the calibration targets. Calibration itself, which
fits the FIC values to those targets, is a separate process that reads them.

These rules describe the end state that the reorganization
(`data-reorganization-plan.md`) works toward, so reviewers can agree on it before
anything moves. Where today's pipeline differs, the phase that brings it into line is
named. Decisions, with their dates, are in
[decisions.md](decisions.md).

Items marked **Proposed** are not yet decided. Each one needs a ruling before the
phase that applies it.

The technology database follows the same rules and has its own detailed design in
[`technology_database/docs/design.md`](../technology_database/docs/design.md).

## 1. Names

**Lowercase kebab-snake throughout, wherever possible** (D4, D5). A hyphen `-` joins
the words within a part; an underscore `_` separates the parts.

| What | Today | After the reorganization |
| - | - | - |
| Folders and data files | `iron_and_steel_ab.csv`, `CIMS_base_AB.csv`, `FIC_AB.csv` | `iron-and-steel_ab.csv`, `cims-base_ab.csv`, `fixed-intangible-cost_ab.csv` |
| Extracted files | as named | `author_year_slug[_suffix].csv`, as in the technology database: `cer_2026_benchmark-price.csv` |
| Column headers | `Branch`, `Sub_Context` | `branch`, `sub-context` |
| Branch paths | `CIMS.CAN.BC.Iron and Steel` | `cims.can.bc.iron-and-steel` |
| Regions | `BC`, `RoW` | `bc`, `rest-of-world` |
| Sectors, fuels, technologies | `Iron and Steel`, `Natural Gas` | `iron-and-steel`, `natural-gas` |
| Parameters | `service_request`, `market_share_new_max` | `service-request`, `market-share-new-max` |
| Units | `2025_CAD/GJ`, `tCO2e` | `2025_cad/gj`, `t-co2e` (§1.1) |
| Python files and packages | `Reference.py`, `CIMS_Functions/`, `paramLoc.py` | underscores only, lowercase: `reference.py`, `cims_functions/`, `param_loc.py` |

- **Python is the exception.** Module and package names use underscores only, because
  Python can't import a hyphenated name.
- **Exempt:** conventional file names (`README.md`, `LICENSE`, `CITATION.cff`,
  `uv.lock`, `pyproject.toml`) and the `CIMS` import name.
- **Display spellings** ("Iron and Steel", "BC") come from `config/display-name.csv`
  and are used only when results are shown or reported.
- **When each part changes.** Folders and data file names change in phase 3, the
  pipeline's own code in phases 4 to 8, and the Python file names in phase 11. The
  values the model reads (headers, branch paths, regions, parameters, and units)
  change in phase 14. That is a model-core change, with its own pull requests and a
  transition period in which the model reads both spellings.

### 1.1 Units (Proposed)

Lowercasing units loses the case that SI prefixes depend on (`Mt` and `mt`, `MW` and
`mW`). So units come from a closed list rather than a spelling rule alone:

- `config/vocabulary/unit.csv` lists every unit the pipeline may write, with its
  display spelling, its dimension, and its factor to the dimension's base unit. A
  check fails on any unit that isn't listed.
- Within a unit: `/` means "per" (`gj/t`), and `-` joins the words of one quantity
  (`t-co2e`, `vehicle-km`).
- Money is `yyyy_ccc`: the currency year, then the ISO currency code (`2025_cad`,
  `2013_gbp`), with `/` for "per" (`2025_cad/gj`). This matches the technology
  database, where `_` separates the year and the currency.
- A blank unit means "dimensionless" only for parameters listed as dimensionless in
  the vocabulary. Elsewhere it's an error. The exception today is `fic` and
  `dic_min`, whose blank unit means the untagged currency year (D9, decisions.md).

## 2. Data layers

The layers follow the flow of the data (D1). Each layer is written by one kind of
process, and nothing reads a layer upstream of its own place in the flow.

| Layer | In git | Written by | Contents |
| - | - | - | - |
| `data/primary-sources/` | no | hand (downloads) | Documents as obtained, in publisher folders (`gov-can/statcan/`, `gov-usa/eia/`, …). The build never reads them (Q3). |
| `data/extracted/` | no | hand or extraction scripts | As-published values in long CSVs, transcribed by hand or by a script, in the same publisher folders, with one manifest for all sources. Deposited to FRDR and fetched at a pinned version. Source-specific extraction scripts sit beside their CSVs. |
| `data/config/` | yes | hand | Every hand-edited record: the model hierarchy, parameter values not computed from data, mappings, assumptions, proxies, calibration-target recipes, policies, model defaults, and pipeline settings (D2, D3). |
| `data/processed/` | no | the build | One table per source dataset, in the model schema (§3). |
| `data/model/` | no (except its README) | the build | Everything CIMS reads. Nothing in it is edited by hand (Q2). |

- **Choices go only in `config/`.** A value someone chose (a proxy region, a growth
  rate, a fuel assumption) is a record in `config/` with its rationale, not a line of
  code or a cell in a generated file. Hand work in `extracted/` is transcription only:
  values as the source publishes them.
- **Assumptions from an analysis (Proposed).** Where an assumption is the result of a
  small analysis, its marimo notebook and the values it writes sit together in
  `config/assumption/<topic>/`: `<topic>.csv` (or another file the build can read),
  `<topic>_analysis.py`, and a README entry with the rationale and the date. The build
  reads only the values and never runs the notebook, so changing the assumption is a
  deliberate act: rerun the notebook and commit the new values. The notebook records
  which inputs it used (files and dates), because the values don't update when those
  inputs change. An analysis that should rerun whenever its inputs change is a stage
  instead, and writes to `processed/`.
- **Generated files are never committed.** They always come from the build (Q2).
- **The data folder follows the code.** The pipeline reads and writes the `data/`
  folder beside the code it imports, unless `CIMS_DATA_ROOT` is set. Run it with
  `uv run` from the worktree. Phase 2 puts every layer's root in one module,
  `common/paths.py`.
- **Today's folders** map to the layers this way (phase 3 moves them, and phase 13
  retires `fixed_data`):
  - `raw_data/` to `extracted/`, and its documents to `primary-sources/`
  - `mappings_conversions/`, `technology_database/`, and the hand-made parts of
    `model_inputs/` to `config/`
  - `processed_data/` to `processed/`
  - the generated parts of `model_inputs/`, and `calibration/`, to `model/`

## 3. Schema

Processed tables and model files share one schema (Q1): the CIMS model-description
columns, in this order:

`branch, type, region, sector, service, technology, parameter, context, sub-context, target, source, unit, year, value`

- **One vocabulary.** Values follow §1: branch paths, region codes, fuel names, and
  parameter names in lowercase kebab-snake, and money units as `yyyy_ccc`.
- **Columns the build can derive** (`type`, `region`, `sector`, and `service` all
  follow from `branch`) may be left blank in processed tables. The model stage fills
  them.
- **Values that aren't model parameters.** Some processed tables hold values that a
  later stage turns into model rows, such as a sector's activity shares or a growth
  rate. Today these tables have their own columns (`Variable`, for example, with
  values like `Metal Smelting.Aluminum`). After the reorganization, they use the same columns as
  every other table, with a `parameter` name such as `activity-share` or
  `growth-rate`. Each such name is listed, with its definition and unit, in
  `config/vocabulary/processed-parameter.csv`. A check then accepts only model
  parameters and listed names, so a misspelled parameter can't slip through.
- **Years.** A blank `year` is a constant for every year. Keep only the anchor years
  for parameters the model interpolates, and collapse a series that never changes to
  one blank-year row. The model fills years only strictly between the years given,
  and never for a parameter that also has a constant.

## 4. CSV files

- **Encoding: UTF-8 with a byte-order mark**, so Excel reads accents and symbols. The
  technology database already writes this way (Brad, 2026-11-06). Readers accept files
  with or without the mark. **Proposed** for model files, which have no mark today.
  The model reader and the validator (both polars) and pandas skip the mark, but
  Python's `csv` module opened with `encoding="utf-8"` keeps it in the first header,
  so code that reads model files that way (in notebooks or emh's reporting) needs
  `utf-8-sig` first.
- **Line endings: `\n`**, whatever the platform. The repository's `.gitattributes`
  sets `eol=lf`.
- **Comma-separated, one header row**, with the columns in schema order (§3).
- **Missing values are empty cells.** Never write `nan`, `NaN`, `None`, or `null`. The
  string `none` is a model value meaning "no value" and stays as written.
- **No index column.** With pandas, write with `index=False`.
- **Rows in a stable order**, so two runs of the same code give the same bytes. Sort
  by the key columns where order carries no meaning.
- **Phase 2 adds `common/io.py`** to read and write CSVs this way, and every stage
  uses it.

## 5. Precedence between files

When two rows set the same parameter on the same branch, today's model mostly keeps the
last one loaded, so file order decides which row wins. There are exceptions:

- a constant (blank year) never beats a year row;
- a blank value never overwrites (the string `none` does);
- node constants (`competition`, `is_supply`, `aggregate_structural`) raise an error
  instead of overwriting;
- region and sector keep the first value, not the last.

**Make precedence explicit; don't rely on file order.** A file that overrides others
on purpose (such as the Natural Gas Market file) is declared as an override in the
stage table, and phase 6's `--check` flags any other key that appears in more than one
file. Then files can be merged or reordered freely, and the snapshot comparison (§8)
shows any effect. Validators read all files concatenated, before overrides, so a
check that sums rows must allow for overridden values.

## 6. Code

- **One home for each kind of code.** Generic code (paths, configuration, CSV input
  and output, the schema, branch handling, years, currency, shared checks) goes in
  `common/`; code for one dataset goes in `source/<publisher>/`; model assembly in
  `model/`; the calibration targets in `calibration_target/` (named apart from
  `src/CIMS/calibration/`, the FIC fitting); and the few sectors that need their
  own code in `hooks/`.
- **Stages read files.** A stage reads files and writes files. It doesn't import
  another stage to get its data, and no script that builds calibration targets reads
  another's output (phase 6).
- **Absolute imports** (`from CIMS.data_processing.utils.currency import ...`), and no
  `sys.path` edits.
- **No data paths in code** outside `common/paths.py`, and no settings in code:
  years, currency settings, and scenario choices go in `config/pipeline.toml`
  (phase 4).
- **Fail loudly.** A missing input, an unknown name, or a unit that doesn't convert
  stops the build with a message naming the file and the row. A warning is for data
  that is usable but worth a look.

## 7. Documentation

- **Package docs** in this folder: `README.md`; `design.md` (the rules);
  `pipeline.md` (the stage table, and how to add a source, a sector, or a region,
  change a proxy, and deposit to FRDR); `outputs.md` (every file in `processed/` and
  `model/`, and what it's for); `decisions.md`; and this file.
- **Module docstrings** follow one template:

  ```
  """One-line summary.

  Reads
  -----
  <layer>/<path>: what is used from it

  Writes
  ------
  <layer>/<path>: schema, one row per …

  Method
  ------
  Numbered steps, each naming its assumption or the decision in docs/decisions.md.

  Usage
  -----
  python -m …
  """
  ```

  A decision goes in `decisions.md`, with its date and record, and the docstring
  points to it. Undated "Decisions/notes" lists in docstrings move there (phase 10).
- **Function docstrings** in NumPy style, on every public function. Once phase 10 is
  done, a check counts missing docstrings and fails above zero.
- **READMEs.** Each `config/` file and each `extracted/<publisher>/` folder has an
  entry in a folder table, as in the technology database. `data/README.md` describes
  the five layers.
- **Writing.** Plain language and the active voice; the Oxford comma; Canadian Press
  spelling.

## 8. Commits and pull requests

- **Separate moves from edits.** A move commit is `git mv` plus the path-string
  updates it forces, and nothing else. A refactor commit changes no paths.
- **Report every commit's effect on outputs** with `tools/output_snapshot.py`:

  ```
  uv run python tools/output_snapshot.py compare baseline
  uv run python tools/output_snapshot.py compare baseline --rename tools/renames.csv
  ```

  A commit that only restructures says so in its message: "outputs identical:
  snapshot <id>". A commit that changes outputs on purpose stands alone, says what
  changed and why, summarizes the comparison, and adds an entry to `decisions.md`
  (D10).
- **Ignored data moves with a script.** Git doesn't move ignored files, so each layout
  move ships with `tools/migrate_data_layout.py`, which moves an existing local
  `data/` to the new layout and can be run more than once.
- **Size.** One concern per commit, ideally under about 400 changed lines of real edits
  (moves don't count). One pull request per phase into `staging` (D7).
- **Message shape:** `<area>: <what>`, a blank line, why, and the outputs line. For
  example: `data_processing/calibration_target: replace 18 sector modules with
  targets.py and target-recipe.csv`.
- **Stage files by name.** Never `git add -A`, which would pick up untracked results.
