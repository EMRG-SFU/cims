# technology_database

The build of the industrial technology database: archetypes of the technologies that can
serve each industrial end use in CIMS, summarised from published sources. The rules are in
[docs/design.md](docs/design.md), the build and how to add a source in
[docs/pipeline.md](docs/pipeline.md), and every decision in [docs/decisions.md](docs/decisions.md).

CIMS is this package's home (since 2026-11-20, plan D6): change the code and the records here.
The technology-database repository (brad-griffin/technology-database) keeps the extractions'
authoring copy and the scripts that make them, the reviews, the review packages, and source
intake; it installs CIMS to use this package. Its history up to the move is there (`97a29cf`).

```
python -m CIMS.data_processing.technology_database.build            # everything; fails on any broken rule
python -m CIMS.data_processing.technology_database.build --check    # the checks only
```

Its data folders come from `common/paths.py`, like every other stage, so the working directory
does not matter.

| Folder (from the repo root) | Edited by | Contents |
| - | - | - |
| `data/extracted/technology-database/` | extraction | Layer 1: the source manifest (`source-details_*.csv`) and every extraction CSV, in publisher folders. Copied from the technology-database repository's authoring copy until the FRDR deposit, then downloaded from FRDR; not committed in CIMS. |
| `data/config/technology-database/` | hand | The records: `archetype.csv`, `archetype-service.csv`, `source-example.csv`, `family.csv`, `hierarchy-annotation.csv`, the CIMS hierarchy snapshot, `fill/`, `reference/` and `vocabulary/`. |
| `data/processed/technology-database/` | the build | The example and archetype tables, the reviews, `unfilled/`, `vocabulary.md` and the workbook. Never edit these: the build rewrites them. [docs/outputs.md](docs/outputs.md) says what each file is for. |
| `primary-sources/` | hand | The source documents, in the same publisher folders as the extractions. Local only; the build and the checks work without it. |

| Code | What it is |
| - | - |
| `build.py`, `pipeline/` | The build, stage by stage (docs/pipeline.md). |
| `extraction/check_manifest.py` | The source manifest against the disk and the database. |
| `extraction/check_names.py` | The names in the extraction layer against the naming rule. |
| `extraction/register_source.py` | Registers one extracted source in the manifest and the records. |
| `extraction/input_template.py` | A marimo app for entering a source's values in the extraction format. |
| `extraction/paths.py` | Where the extractions, documents, records, and outputs are, for the scripts above. |

The scripts written for single sources (workbooks, bulk downloads, PDF tables) are not in the
package: each sits beside its CSVs in `data/extracted/technology-database/<publisher>/`, so it
goes to FRDR with them, and reads its document from the same folder under `primary-sources/`.

To change what an example is an example of, edit its row in
`data/config/technology-database/source-example.csv`. For everything else, see the *How to*
section of the pipeline doc.
