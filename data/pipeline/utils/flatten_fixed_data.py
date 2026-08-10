import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.data_extensions import extend_series_linear, extend_series_trend_decline, extend_constant
from utils.data_fill import interpolate_5year_to_annual, backfill_constant

# Usage:
# 1. Open a command prompt in the folder containing this script.
# 2. Run the script with the input folder and output folder paths.
#    Example:
#      python flatten_year_columns.py \
#        "C:\cims\data\raw_data\fixed_data\Agriculture" \
#        "C:\Other Scripts" \
#        --year-min 2000 --year-max 2050 --target-start 2000 --target-end 2100 --target-step 1
#
# 3. Or call main() directly from Python if you prefer programmatic use:
#      from flatten_year_columns import main
#      main(
#          input_folder=r"C:\cims\data\raw_data\fixed_data\Agriculture",
#          output_folder=r"C:\Other Scripts",
#          year_min=2000,
#          year_max=2050,
#          target_start=2000,
#          target_end=2100,
#          target_step=1,
#      )
#
# 4. The script reads every CSV in the input folder and all its subfolders.
# 5. It finds source year columns between 2000 and 2050, drops Comments, and
#    writes annual rows for each item between 2000 and 2100.
# 6. Output CSV files are saved under the output folder in a matching subfolder.



# Non-numeric tokens that mean "no value" rather than a real literal flag
# (e.g. "-" placeholders for zero/blank cost cells), so they should still
# collapse to a blank row instead of being carried forward like "FALSE"/"Myopic".
BLANK_LITERAL_SENTINELS = {"-", "--", "n/a", "na"}

# Parameters whose classification actually lives in the Context column in the
# raw fixed-data CSVs (e.g. competition's "Sector"/"Fixed Ratio"/"Tech
# Compete"). When such a row has no year-column value at all, move that
# Context content into Value instead, so it lines up with every other
# Parameter's convention of storing its content in Value.
CONTEXT_TO_VALUE_PARAMETERS = {"competition", "market_share_class", "dcc_class", "dic_class"}


def parse_year_columns(header, year_min=2000, year_max=2050):
    """Identify year columns in the CSV header.

    This scans the header row for columns whose labels represent numeric years.
    It handles labels like "2000", "2,000.00", and similar formatting.
    """
    year_columns = []
    for idx, label in enumerate(header):
        normalized = label.strip().replace(",", "")
        if normalized.endswith(".00"):
            normalized = normalized[:-3]
        if normalized.isdigit():
            year = int(normalized)
            if year_min <= year <= year_max:
                year_columns.append((year, idx))
    return sorted(year_columns)


def safe_float(value):
    """Convert a value to float if possible, otherwise return None."""
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    value = value.replace(",", "")
    try:
        return float(value)
    except ValueError:
        return None


def compute_cagr(start_value, end_value, years):
    """Compute annualized linear growth rate between two values."""
    if years <= 0 or start_value is None or end_value is None:
        return 0.0
    if start_value == 0 or pd.isna(start_value) or pd.isna(end_value):
        return 0.0
    try:
        return (end_value / start_value) ** (1.0 / years) - 1.0
    except Exception:
        return 0.0


def open_input_file(input_path):
    """Open a CSV file with fallback encodings for non-UTF-8 input files."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return input_path.open(newline="", encoding=encoding)
        except UnicodeDecodeError:
            continue
    return input_path.open(newline="", encoding="latin-1", errors="replace")


def process_file(input_path, output_path, year_min, year_max, target_start, target_end, target_step):
    """Read one CSV, flatten its year columns, and write transformed output."""
    encodings_to_try = ("utf-8-sig", "utf-8", "latin-1")
    last_error = None

    for encoding in encodings_to_try:
        try:
            with input_path.open(newline="", encoding=encoding) as infile:
                reader = csv.reader(infile)
                header = next(reader, None)
                if header is None:
                    raise ValueError(f"File {input_path} is empty")

                # Identify the columns that represent the source year values.
                year_columns = parse_year_columns(header, year_min, year_max)
                if not year_columns:
                    raise ValueError(f"No year columns found in {input_path} between {year_min} and {year_max}")

                # Locate the 2000 source column and the Parameter column for filtering.
                year_2000_idx = next((idx for year, idx in year_columns if year == 2000), None)
                normalized_header = [h.strip().lower().replace("_", "").replace(" ", "") for h in header]
                parameter_index = next(
                    (i for i, name in enumerate(normalized_header) if name in ("parameter", "paramater")),
                    None,
                )
                comments_index = next(
                    (i for i, name in enumerate(normalized_header) if name == "comments"),
                    None,
                )
                context_index = next(
                    (i for i, name in enumerate(normalized_header) if name == "context"),
                    None,
                )

                # Preserve non-year columns for output, but drop the Comments column.
                year_header_indexes = [idx for _, idx in year_columns]
                non_year_indexes = [
                    idx
                    for idx in range(len(header))
                    if idx not in year_header_indexes and idx != comments_index
                ]
                non_year_headers = [header[idx].strip() for idx in non_year_indexes]
                output_headers = non_year_headers + ["Year", "Value"]
                context_output_pos = (
                    non_year_indexes.index(context_index)
                    if context_index is not None and context_index in non_year_indexes
                    else None
                )

                target_years = list(range(target_start, target_end + 1, target_step))
                rows_written = 0

                with output_path.open("w", newline="", encoding="utf-8") as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(output_headers)

                    for line_number, row in enumerate(reader, start=2):
                        if not any(cell.strip() for cell in row):
                            continue

                        row = [cell.strip() for cell in row]

                        # If this row is market_share_total, keep only year 2000 and skip it if 2000 is missing.
                        is_market_share_total = False
                        market_share_2000_value = None
                        if parameter_index is not None and parameter_index < len(row):
                            if row[parameter_index].strip().lower() == "market_share_total":
                                is_market_share_total = True
                                if year_2000_idx is not None:
                                    raw_2000 = row[year_2000_idx] if year_2000_idx < len(row) else ""
                                    market_share_2000_value = safe_float(raw_2000)
                                if market_share_2000_value is None:
                                    continue

                        # Collect every year column that actually has a value, keeping
                        # numeric values as floats and everything else as a literal
                        # string. Sentinel tokens like "-" mean "no value", not a literal.
                        points: list[tuple[int, float | str]] = []
                        for year, idx in year_columns:
                            raw_value = row[idx] if idx < len(row) else ""
                            raw_value = raw_value.strip()
                            if raw_value == "" or raw_value.lower() in BLANK_LITERAL_SENTINELS:
                                continue
                            parsed_value = safe_float(raw_value)
                            points.append((year, parsed_value if parsed_value is not None else raw_value))
                        points.sort(key=lambda p: p[0])

                        # Nothing specified anywhere in the year columns. For a
                        # handful of Parameters, the real classification value
                        # lives in Context instead -- move it into Value here.
                        if not points:
                            output_cells = [row[idx] for idx in non_year_indexes]
                            value = ""
                            if (
                                context_output_pos is not None
                                and parameter_index is not None
                                and parameter_index < len(row)
                                and row[parameter_index].strip().lower() in CONTEXT_TO_VALUE_PARAMETERS
                                and output_cells[context_output_pos] != ""
                            ):
                                value = output_cells[context_output_pos]
                                output_cells[context_output_pos] = ""
                            writer.writerow(output_cells + ["", value])
                            rows_written += 1
                            continue

                        if is_market_share_total:
                            writer.writerow([row[idx] for idx in non_year_indexes] + [2000, market_share_2000_value])
                            rows_written += 1
                            continue

                        distinct_values = {value for _, value in points}

                        # Constant across every year with data (numeric or literal,
                        # including a single known point) -- one row, no Year needed.
                        if len(distinct_values) == 1:
                            writer.writerow([row[idx] for idx in non_year_indexes] + ["", points[0][1]])
                            rows_written += 1
                            continue

                        has_literal = any(isinstance(value, str) for _, value in points)

                        if not has_literal:
                            # Genuine numeric time series -- interpolate/extend as before.
                            known_points = points
                            series = pd.Series({year: value for year, value in known_points}, dtype=float)
                            annual_series = interpolate_5year_to_annual(series)

                            if target_start < annual_series.index.min():
                                annual_series = backfill_constant(annual_series, start_year=target_start)

                            if target_end > annual_series.index.max():
                                if 2045 in annual_series.index and 2050 in annual_series.index:
                                    start_value = annual_series.loc[2045]
                                    end_value = annual_series.loc[2050]
                                    if start_value != end_value:
                                        growth_rate = compute_cagr(float(start_value), float(end_value), 5)
                                        annual_series = extend_series_linear(
                                            annual_series,
                                            base_year=2050,
                                            periods=[(2051, target_end + 1, growth_rate)],
                                        )
                                    else:
                                        annual_series = extend_constant(annual_series, end_year=target_end)
                                else:
                                    annual_series = extend_constant(annual_series, end_year=target_end)

                            annual_series = annual_series.reindex(target_years)
                            for year in target_years:
                                value = annual_series.loc[year]
                                if pd.isna(value):
                                    value = ""
                                writer.writerow([row[idx] for idx in non_year_indexes] + [year, value])
                                rows_written += 1
                            continue

                        # Non-numeric values that vary by year: write the most common
                        # value as the default (blank Year), then an explicit override
                        # row only for each year whose value differs from that default.
                        counts = Counter(value for _, value in points)
                        max_count = max(counts.values())

                        # No value repeats -- there's no real "default" to collapse
                        # to (every year's value is distinct), so keep every row
                        # with its explicit Year instead of arbitrarily blanking
                        # the earliest one.
                        if max_count == 1:
                            for year, value in points:
                                writer.writerow([row[idx] for idx in non_year_indexes] + [year, value])
                                rows_written += 1
                            continue

                        default_value = next(value for _, value in points if counts[value] == max_count)

                        writer.writerow([row[idx] for idx in non_year_indexes] + ["", default_value])
                        rows_written += 1
                        for year, value in points:
                            if value != default_value:
                                writer.writerow([row[idx] for idx in non_year_indexes] + [year, value])
                                rows_written += 1

                return rows_written
        except UnicodeDecodeError as e:
            last_error = e
            if output_path.exists():
                output_path.unlink()
            continue

    raise last_error


def discover_csv_files(folder):
    """Return a sorted list of CSV files in the input folder and all subfolders."""
    return sorted(folder.rglob("*.csv"))


def main(
    input_folder=None,
    output_folder=None,
    year_min=2000,
    year_max=2050,
    target_start=2000,
    target_end=2100,
    target_step=1,
):
    """Entry point for both Python API use and the command line."""
    if input_folder is None or output_folder is None:
        parser = argparse.ArgumentParser(description="Flatten wide year columns and expand data to annual steps.")
        parser.add_argument("input_folder", type=Path, help="Input folder containing CSV files.")
        parser.add_argument("output_folder", type=Path, help="Base output folder for flattened CSV files.")
        parser.add_argument("--year-min", type=int, default=year_min, help="Start year for source columns to flatten.")
        parser.add_argument("--year-max", type=int, default=year_max, help="End year for source columns to flatten.")
        parser.add_argument("--target-start", type=int, default=target_start, help="Start year for annual output.")
        parser.add_argument("--target-end", type=int, default=target_end, help="End year for annual output.")
        parser.add_argument("--target-step", type=int, default=target_step, help="Year step for output rows.")
        args = parser.parse_args()

        input_folder = args.input_folder
        output_folder = args.output_folder
        year_min = args.year_min
        year_max = args.year_max
        target_start = args.target_start
        target_end = args.target_end
        target_step = args.target_step
    else:
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)

    if not input_folder.exists() or not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    final_output_folder = output_folder
    final_output_folder.mkdir(parents=True, exist_ok=True)

    input_files = discover_csv_files(input_folder)
    if not input_files:
        raise FileNotFoundError(f"No CSV files found in {input_folder}")

    for input_file in input_files:
        relative_dir = input_file.parent.relative_to(input_folder)
        output_dir = final_output_folder / relative_dir if relative_dir != Path('.') else final_output_folder
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{input_file.stem}.csv"
        processed_rows = process_file(
            input_file,
            output_file,
            year_min=year_min,
            year_max=year_max,
            target_start=target_start,
            target_end=target_end,
            target_step=target_step,
        )
        print(f"Processed {input_file.name}: {processed_rows} rows -> {output_file.relative_to(output_folder)}")


if __name__ == "__main__":
    main()
