import pandas as pd
import polars as pl
import numpy as np
import textwrap
from typing import List
import time 

from .validation_utils import get_providers, get_requested
from ..utils.model_description import column_list as COL

from .registry import REGISTRY, resolve_kwargs, Phase, Severity


class ModelValidator:
    def __init__(self, csv_file_paths, col_list, year_list, sector_list,
                 csv_update_file_paths=None, default_values_csv_path=None, node_col=COL.branch,
                 target_col=COL.target, root_node="CIMS", list_csv_path=None):

        self.csv_files = csv_file_paths
        self.scenario_files = csv_update_file_paths or []

        self.default_param_df = self.get_default_df(default_values_csv_path)
        self.competition_types = self._get_list(list_csv_path, column_identifier="Competition")

        self.node_col = node_col
        self.target_col = target_col
        self.col_list = col_list
        self.year_list = [str(x) for x in year_list]
        self.sector_list = sector_list

        self.model_df = self._get_model_df()
        self.root_node = root_node

        self.warnings = {}

        self.index2branch_map = self._create_index_to_branch_map()
        self.branch2node_index_map = self._create_branch_to_node_index_map()

        # Track validation runs and whether errors were detected.
        self.file_validation_ran = False
        self.file_validation_errors = False
        self.file_validation_warnings = False
        self.graph_validation_ran = False
        self.graph_validation_errors = False
        self.graph_validation_warnings = False

    def _get_model_df(self, read_base_file=True, read_scenario_files=True):
        files_to_read = []
        if read_base_file:
            for file in self.csv_files:
                files_to_read.append(file)
        if read_scenario_files:
            for file in self.scenario_files:
                files_to_read.append(file)

        appended_data = []
        for csv_file in files_to_read:
            try:
                sheet_df = pl.read_csv(
                    csv_file,
                    use_pyarrow=False,
                    infer_schema_length=0,
                ).to_pandas().replace({np.nan: None, "": None})
                appended_data.append(sheet_df)

            except ValueError:
                print(f"Warning: Unable to parse csv_path at {csv_file}. Skipping.")

        model_df = pd.concat(appended_data,
                             ignore_index=True)  # Add province sheets together and re-index
        
        # Filter sectors (if applicable)
        if self.sector_list:
            if None not in self.sector_list:
                self.sector_list.append(None)
            model_df = model_df[model_df[COL.sector].isin(self.sector_list)]

        meta_cols = [c for c in model_df.columns if c not in ("Year", "Value") and c in self.col_list]
        year_mask = model_df["Year"].isin(self.year_list) | model_df["Year"].isna()
        mdf = model_df[year_mask][meta_cols + ["Year", "Value"]]
        mdf[COL.parameter] = mdf[COL.parameter].str.lower()

        return mdf

    def get_default_df(self, default_values_csv_path):
        if default_values_csv_path is None:
            return pd.DataFrame()

        # Read model_description from excel
        mixed_type_columns = [COL.default_value]
        df = pl.read_csv(
            default_values_csv_path,
            use_pyarrow=False,
            infer_schema_length=0,
            ).with_columns(pl.all().replace(
                {np.nan: None}
            )).to_pandas()

        # Remove empty rows
        df = df.dropna(axis=0, how="all")

        # Convert parameter strings to lower case
        df[COL.parameter] = df[COL.parameter].str.lower()

        # Return
        return df

    def _get_list(self, list_csv_path, column_identifier):
        if list_csv_path is None:
            return []

        # Read List File from CSV
        df = pl.read_csv(
            list_csv_path, 
            use_pyarrow=False,
            infer_schema_length=0).to_pandas()
        
        # Remove empty rows
        df = df.dropna(axis=0, how='all')

        # Extract inheritable parameters
        list_clean = df[column_identifier].str.lower()

        return list_clean    
    
    def _create_branch_to_node_index_map(self):
        branch_index = {b: i for i, b in self.model_df[self.node_col].drop_duplicates(keep='first').items()}
        index_to_node_index_map = {self.index2branch_map[i]: branch_index[self.index2branch_map[i]] for i in self.model_df.index}
        return index_to_node_index_map

    def _create_index_to_branch_map(self):
        return {i: self.model_df[COL.branch].loc[i] for i in self.model_df.index}

    def _raise_concerns(self, concerns: List[object], concern_key: str, concern_desc: str) -> bool:
        """
        Format and print a single validation result.

        Prints the number of issues found, a short description, and (if applicable)
        a pointer to `ModelValidator.warnings[...]`. Messages are wrapped with a
        hanging indent for readability. Nothing unless issues are detected. 
        Returns True when concerns are present.
        """
        if len(concerns) <= 0:
            more_info = ""
        else:
            more_info = f"See ModelValidator.warnings['{concern_key}'] for more info."

        info_str = f"{len(concerns):5} {concern_desc}. {more_info}"

        if len(concerns) > 0:
            wrapped_print = textwrap.fill(
                info_str, 
                width=100, 
                initial_indent="",
                subsequent_indent=" " * (5 + 1)
            )
            print(wrapped_print)
            return True
        return False

    def _run_check(self, check_function, **kwargs) -> bool:
        concern_list, concern_desc = check_function(**kwargs)

        # None signals the check was intentionally skipped
        if concern_list is None:
            print(f" SKIP  {check_function.__name__} — {concern_desc}")
            self.warnings[check_function.__name__] = []
            return False

        has_concerns = self._raise_concerns(
            concern_list, check_function.__name__, concern_desc
        )
        self.warnings[check_function.__name__] = concern_list
        return has_concerns

    def validate(self):
        """
        Backwards-compatible alias for validate_files().
        """
        return self.validate_files()

    def validate_files(self):
        """
        Run file validation checks using the central registry.
        """
        start = time.time()
        
        print("\n=== Validating model files ===")
        
        # Per-run context values that may be reused by multiple checks
        providers = get_providers(self.model_df, self.node_col)
        requested = get_requested(self.model_df, self.target_col)
        context = {
            'providers': providers, 
            'requested': requested
        }

        # ---- Errors ----
        print("\n-- Errors --")
        errors_found = False
        for name, spec in REGISTRY.iter(phase=Phase.FILE, severity=Severity.ERROR):
            kwargs = resolve_kwargs(self, spec.argmap, context)
            errors_found |= self._run_check(spec.fn, **kwargs)
        self.file_validation_errors = errors_found
        if not errors_found:
            print("No errors found!")
        
        # ---- Warnings ----
        print("\n-- Warnings --")
        warnings_found = False
        for name, spec in REGISTRY.iter(phase=Phase.FILE, severity=Severity.WARNING):
            kwargs = resolve_kwargs(self, spec.argmap, context)
            warnings_found |= self._run_check(spec.fn, **kwargs)
        self.file_validation_warnings = warnings_found
        if not warnings_found:
            print("No warnings found!")
        
        timing = f" (completed in {time.time() - start:.2f}s)"
        print(f"\n=== File validation complete{timing} ===")

        self.file_validation_ran = True
   
    def validate_graph(self):
        """
        Run graph-phase validation checks using the central registry.
        """
        start = time.time()
        
        print("\n=== Running graph-phase validation ===")

        context = {}
        # ---- Errors ----
        print("\n-- Errors --")
        errors_found = False
        for name, spec in REGISTRY.iter(phase=Phase.GRAPH, severity=Severity.ERROR):
            kwargs = resolve_kwargs(self, spec.argmap, context)
            errors_found |= self._run_check(spec.fn, **kwargs)
        self.graph_validation_errors = errors_found
        if not errors_found:
            print("No errors found!")
        
        # ---- Warnings ----
        print("\n-- Warnings --")
        warnings_found = False
        for name, spec in REGISTRY.iter(phase=Phase.GRAPH, severity=Severity.WARNING):
            kwargs = resolve_kwargs(self, spec.argmap, context)
            warnings_found |= self._run_check(spec.fn, **kwargs)
        self.graph_validation_warnings = warnings_found
        if not warnings_found:
            print("No warnings found!")    
        
        timing = f" (completed in {time.time() - start:.2f}s)"
        print(f"\n=== Completed graph-phase validation{timing} ===")

        self.graph_validation_ran = True

    def explain(self, key: str = None) -> None:
        """
        Print documentation for validation checks.

        explain()        — summary table of all checks with current counts
        explain("key")   — full documentation for a specific check
        """
        if key is not None:
            self._explain_one(key)
        else:
            self._explain_all()

    def _explain_one(self, key: str) -> None:
        try:
            spec = REGISTRY.get(key)
        except KeyError:
            print(f"Unknown check: '{key}'. Call explain() with no arguments to see all checks.")
            return

        width = 60
        title = key.replace("_", " ").upper()
        print(f"\n{title} ({spec.severity})")
        print("=" * width)

        # What it checks — first paragraph of the docstring
        if spec.fn.__doc__:
            first_para = []
            for line in spec.fn.__doc__.strip().splitlines():
                stripped = line.strip()
                if not stripped and first_para:
                    break
                if stripped:
                    first_para.append(stripped)
            print("\n" + " ".join(first_para))

        # Output format and interpretation
        if spec.help_text:
            print()
            for line in spec.help_text.splitlines():
                print(f"  {line}" if line else "")
        else:
            print("\n  (No additional output documentation yet.)")

        # Current result
        print()
        if key in self.warnings:
            count = len(self.warnings[key])
            if count == 0:
                print("  Current result: no issues found.")
            else:
                print(f"  Current result: {count} issue(s) found.")
                print(f'  To inspect:     validator.warnings["{key}"]')
        else:
            print("  (Validation has not been run yet for this check.)")

    def _explain_all(self) -> None:
        col_w = 42

        for phase in [Phase.FILE, Phase.GRAPH]:
            phase_checks = list(REGISTRY.iter(phase=phase))
            if not phase_checks:
                continue

            print(f"\n{'─' * 70}")
            print(f"  {phase.upper()} PHASE CHECKS")
            print(f"{'─' * 70}")

            for severity_label, severity in [("Errors", Severity.ERROR), ("Warnings", Severity.WARNING)]:
                checks = [(n, s) for n, s in phase_checks if s.severity == severity]
                if not checks:
                    continue

                print(f"\n  {severity_label}")
                print(f"  {'Check':<{col_w}} {'Count':>6}  Description")
                print(f"  {'-' * (col_w + 30)}")

                for name, spec in checks:
                    count_str = ""
                    if name in self.warnings:
                        count_str = str(len(self.warnings[name]))

                    if spec.short_desc:
                        doc_line = spec.short_desc
                    elif spec.fn.__doc__:
                        doc_line = spec.fn.__doc__.strip().splitlines()[0].strip()
                    else:
                        doc_line = ""

                    print(f"  {name:<{col_w}} {count_str:>6}  {doc_line}")

        print(f"\n{'─' * 70}")
        print("  Call explain('check_name') for full documentation on any check.")
