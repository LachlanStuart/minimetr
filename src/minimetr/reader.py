# Reader implementation will go here

import sqlite3
import json
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Tuple, Set, Union
from collections import namedtuple
import itertools

# Define the structure for pivot results
PivotResult = namedtuple(
    "PivotResult", ["index_tuples", "column_tuples", "values_array"]
)


class Reader:
    """Reads metrics from a minimetr SQLite database."""

    def __init__(self, db_path: str):
        """Initializes the reader and database connection."""
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        # Caches for faster lookups during read
        self._session_cache: Dict[int, Dict[str, Any]] = {}
        self._step_cache: Dict[int, Dict[str, Any]] = {}
        self._step_session_map: Dict[int, int] = {}  # Map step_id -> session_id
        self._metric_def_cache: Dict[int, Dict[str, Any]] = {}
        self._metric_set_cache: Dict[int, Tuple[int, ...]] = {}
        self._keys_cache: Optional[Dict[str, List[str]]] = None
        # print(f"Reader initialized for {db_path}")
        self._get_db_connection()  # Establish connection on init

    def _get_db_connection(self):
        # Placeholder: Returns a connection
        if self._conn is None:
            # Readers should use immutable=1 for potential performance benefits
            # and protection against accidental writes through the reader connection.
            uri = f"file:{self._db_path}?mode=ro&immutable=1"
            try:
                # Use URI=True for mode=ro support
                self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            except sqlite3.OperationalError:
                # Fallback for systems where URI mode might not be fully supported
                # or if the file must be writable for some reason (e.g., WAL mode without immutable)
                print(
                    "[Warning] Could not open DB in read-only immutable mode. Falling back."
                )
                self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            # Set row factory for dict access
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _load_definitions(self):
        """Load definitions from DB into caches if not already loaded."""
        # Only load if any cache is empty
        if (
            self._session_cache
            and self._step_cache
            and self._metric_def_cache
            and self._metric_set_cache
        ):
            return

        conn = self._get_db_connection()
        cursor = conn.cursor()

        # Load Sessions
        cursor.execute("SELECT session_id, run_info_json FROM Sessions")
        for row in cursor.fetchall():
            self._session_cache[row["session_id"]] = json.loads(row["run_info_json"])

        # Load Steps
        cursor.execute("SELECT step_id, session_id, step_json FROM Steps")
        for row in cursor.fetchall():
            self._step_cache[row["step_id"]] = dict(json.loads(row["step_json"]))
            self._step_session_map[row["step_id"]] = row["session_id"]

        # Load MetricDefinitions
        cursor.execute("SELECT metric_def_id, definition_json FROM MetricDefinitions")
        for row in cursor.fetchall():
            self._metric_def_cache[row["metric_def_id"]] = dict(
                json.loads(row["definition_json"])
            )

        # Load MetricDefinitionSets
        cursor.execute(
            "SELECT definition_set_id, metric_def_ids_json FROM MetricDefinitionSets"
        )
        for row in cursor.fetchall():
            self._metric_set_cache[row["definition_set_id"]] = tuple(
                json.loads(row["metric_def_ids_json"])
            )

        self._keys_cache = None  # Invalidate keys cache

    @property
    def session_ids(self) -> List[int]:
        """Returns a sorted list of all session IDs in the database."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT session_id FROM Sessions ORDER BY session_id")
        return [row["session_id"] for row in cursor.fetchall()]

    def _resolve_session(self, session: Union[int, str, None]) -> Optional[int]:
        """Resolve a session parameter to a concrete session_id.

        - None: no filtering (returns None)
        - "latest": resolves to max(session_id)
        - int: returned as-is
        """
        if session is None:
            return None
        if session == "latest":
            ids = self.session_ids
            if not ids:
                return None
            return ids[-1]
        if isinstance(session, int):
            return session
        raise ValueError(f"session must be int, 'latest', or None, got {session!r}")

    @property
    def keys(self) -> List[str]:
        """Returns a flat, sorted list of all unique keys found across run_info, steps, and metrics."""
        categorized = self._get_categorized_keys()
        all_keys_set: Set[str] = set()
        for key_list in categorized.values():
            all_keys_set.update(key_list)
        return sorted(list(all_keys_set))

    @property
    def run_keys(self) -> List[str]:
        """Returns a sorted list of unique keys found in run_info across all sessions."""
        return self._get_categorized_keys().get("run", [])

    @property
    def step_keys(self) -> List[str]:
        """Returns a sorted list of unique keys found in step definitions."""
        return self._get_categorized_keys().get("step", [])

    @property
    def metric_keys(self) -> List[str]:
        """Returns a sorted list of unique keys found in metric definitions."""
        return self._get_categorized_keys().get("metric", [])

    def _get_categorized_keys(self) -> Dict[str, List[str]]:
        """Internal helper to get categorized keys, using cache."""
        if self._keys_cache is not None:
            return self._keys_cache

        self._load_definitions()  # Ensure caches are populated
        keys_by_type: Dict[str, Set[str]] = {
            "run": set(),
            "step": set(),
            "metric": set(),
        }

        # Infer keys from cached definitions
        for run_info in self._session_cache.values():
            keys_by_type["run"].update(run_info.keys())

        step_keys_found = set()
        for step_def in self._step_cache.values():
            step_keys_found.update(step_def.keys())
        # Assign step keys, ensuring they don't overlap with run keys
        # (this assumes run keys are less likely to be intentionally shadowed by step keys)
        keys_by_type["step"] = step_keys_found - keys_by_type["run"]

        for metric_def in self._metric_def_cache.values():
            # Assign metric keys, ensuring they don't overlap with run or step keys
            keys_by_type["metric"].update(
                metric_def.keys() - keys_by_type["run"] - keys_by_type["step"]
            )

        # Final sorted lists
        self._keys_cache = {k: sorted(list(v)) for k, v in keys_by_type.items()}
        return self._keys_cache

    def read(self, session: Union[int, str, None] = None, **filters) -> List[Dict[str, Any]]:
        """Reads data points, applying filters across all key types.

        Args:
            session: Optional session filter. If "latest", resolves to the most
                recent session_id. If an int, filters to that session_id. If None
                (default), reads all sessions.
            **filters: Key-value or key-callable filters applied to each record.
        """
        self._load_definitions()  # Ensure definition caches are populated
        conn = self._get_db_connection()
        cursor = conn.cursor()

        resolved_session = self._resolve_session(session)
        if resolved_session is not None:
            # Filter at the SQL level by JOINing DataPoints with Steps
            cursor.execute(
                "SELECT dp.step_id, dp.definition_set_id, dp.values_blob "
                "FROM DataPoints dp "
                "JOIN Steps s ON dp.step_id = s.step_id "
                "WHERE s.session_id = ?",
                (resolved_session,),
            )
        else:
            cursor.execute(
                "SELECT dp.step_id, dp.definition_set_id, dp.values_blob FROM DataPoints dp"
            )

        results = []
        lambda_filters = {k: v for k, v in filters.items() if callable(v)}
        static_filters = {k: v for k, v in filters.items() if not callable(v)}

        for row in cursor.fetchall():
            step_id = row["step_id"]
            def_set_id = row["definition_set_id"]
            values_blob = row["values_blob"]

            step_context = self._step_cache.get(step_id)
            session_id = self._step_session_map.get(step_id)
            run_info = (
                self._session_cache.get(session_id) if session_id is not None else None
            )
            metric_def_ids = self._metric_set_cache.get(def_set_id)

            if step_context is None or run_info is None or metric_def_ids is None:
                # print(f"[Warning] Skipping data point due to missing definition: step_id={step_id}, session_id={session_id}, def_set_id={def_set_id}")
                continue

            # Decode values blob
            try:
                values = np.frombuffer(values_blob, dtype=np.float32)
                if len(values) != len(metric_def_ids):
                    # print(f"[Warning] Skipping data point due to value/def mismatch: step_id={step_id}")
                    continue
            except ValueError:
                # print(f"[Warning] Skipping data point due to blob decoding error: step_id={step_id}")
                continue

            # Combine run_info, step context, metric def, and value
            for i, metric_def_id in enumerate(metric_def_ids):
                metric_def = self._metric_def_cache.get(metric_def_id)
                if metric_def is None:
                    # print(f"[Warning] Skipping metric due to missing metric definition: metric_def_id={metric_def_id}")
                    continue

                record = {"value": values[i], **run_info, **step_context, **metric_def}

                # Apply filters
                match = True
                # Check static first
                for key, filter_val in static_filters.items():
                    if record.get(key) != filter_val:  # Use .get for safer check
                        match = False
                        break
                if not match:
                    continue
                # Check lambda
                for key, filter_func in lambda_filters.items():
                    if key not in record or not filter_func(record[key]):
                        match = False
                        break
                if not match:
                    continue

                results.append(record)

        return results

    def pivot(
        self,
        index: List[str],
        columns: List[str],
        filter: Optional[Dict[str, Any]] = None,
        session: Union[int, str, None] = None,
    ) -> PivotResult:
        """Pivots the data based on specified index and column keys.

        Args:
            index: List of keys to use for the pivot table index (rows).
            columns: List of keys to use for the pivot table columns.
            filter: Optional dictionary of key-value pairs or key-lambda pairs
                    to filter the data before pivoting.
            session: Optional session filter (int, "latest", or None).

        Returns:
            A PivotResult namedtuple containing:
            - index_tuples: List of unique index tuples (sorted).
            - column_tuples: List of unique column tuples (sorted).
            - values_array: Numpy array (float32) with pivoted values (NaN where missing).
        """
        data = self.read(session=session, **(filter or {}))
        if not data:
            return PivotResult(
                index_tuples=[],
                column_tuples=[],
                values_array=np.array([]).astype(np.float32),
            )

        # --- Identify unique index and column tuples and create mappings ---
        index_key_tuples: Set[Tuple] = set()
        column_key_tuples: Set[Tuple] = set()
        temp_data_map: Dict[Tuple, Dict[Tuple, float]] = {}

        # Check for missing keys before iterating
        required_keys = set(index) | set(columns)
        if data:
            available_in_data = set(data[0].keys())
        else:
            available_in_data = set()

        missing_keys = required_keys - available_in_data
        if missing_keys:
            all_available_keys = self.keys
            raise KeyError(
                f"Cannot pivot: Key(s) {missing_keys} not found in data. Available keys: {all_available_keys}"
            )

        for record in data:
            try:
                idx_tuple = tuple(record[k] for k in index)
                col_tuple = tuple(record[k] for k in columns)
                value = record["value"]

                index_key_tuples.add(idx_tuple)
                column_key_tuples.add(col_tuple)

                # Store temporarily, handling potential duplicates (e.g., take first)
                if idx_tuple not in temp_data_map:
                    temp_data_map[idx_tuple] = {}
                if (
                    col_tuple not in temp_data_map[idx_tuple]
                ):  # Or decide how to aggregate
                    temp_data_map[idx_tuple][col_tuple] = float(value)  # Ensure float
            except KeyError as e:
                # Should be caught by the initial check, but belt-and-suspenders
                all_available_keys = self.keys
                raise KeyError(
                    f"Error accessing key '{e}' during pivot prep. Record: {record}. Available keys: {all_available_keys}"
                )

        # --- Sort unique tuples and create index/column mappings ---
        sorted_index_tuples = sorted(list(index_key_tuples))
        sorted_column_tuples = sorted(list(column_key_tuples))

        index_map = {tuple_val: i for i, tuple_val in enumerate(sorted_index_tuples)}
        column_map = {tuple_val: j for j, tuple_val in enumerate(sorted_column_tuples)}

        # --- Create and fill the result array ---
        num_rows = len(sorted_index_tuples)
        num_cols = len(sorted_column_tuples)
        values_array = np.full((num_rows, num_cols), np.nan, dtype=np.float32)

        for idx_tuple, col_map_for_idx in temp_data_map.items():
            row_idx = index_map[idx_tuple]
            for col_tuple, value in col_map_for_idx.items():
                if col_tuple in column_map:  # Check if column is actually used
                    col_idx = column_map[col_tuple]
                    values_array[row_idx, col_idx] = value

        return PivotResult(
            index_tuples=sorted_index_tuples,
            column_tuples=sorted_column_tuples,
            values_array=values_array,
        )

    def pandas(
        self,
        index: List[str],
        columns: List[str],
        filter: Optional[Dict[str, Any]] = None,
        index_formatter: Optional[Callable] = None,
        column_formatter: Optional[Callable] = None,
        session: Union[int, str, None] = None,
    ):
        """Returns data as a Pandas DataFrame.

        Args:
            session: Optional session filter (int, "latest", or None).
        """
        # TODO: Implement pandas export
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for .pandas() export. Please install it."
            )

        print(
            f"[Placeholder] Exporting to Pandas: index={index}, columns={columns}, filter={filter}"
        )
        # Basic implementation: Read all matching, then pivot
        data = self.read(session=session, **(filter or {}))
        if not data:
            return pd.DataFrame()  # Return empty DataFrame if no data

        df = pd.DataFrame(data)
        # Ensure 'value' column exists
        if "value" not in df.columns:
            raise ValueError("Cannot pivot: 'value' column missing from data.")

        try:
            # Ensure all index/column keys exist before pivoting
            missing_keys = [k for k in index + columns if k not in df.columns]
            if missing_keys:
                raise KeyError(
                    f"Cannot pivot: Key(s) {missing_keys} not found in data columns {df.columns.tolist()}. Ensure index/columns keys exist in run_info, step, or metric definitions."
                )

            pivot_df = pd.pivot_table(
                df, values="value", index=index, columns=columns, aggfunc="first"
            )  # Assuming one value per index/col combo

            # Apply formatters if provided
            if index_formatter and isinstance(pivot_df.index, pd.MultiIndex):
                pivot_df.index = pivot_df.index.map(index_formatter)
            if column_formatter and isinstance(pivot_df.columns, pd.MultiIndex):
                pivot_df.columns = pivot_df.columns.map(column_formatter)

            return pivot_df
        except KeyError as e:
            raise KeyError(
                f"Error during pivoting: Key {e} likely involved. Data columns: {df.columns.tolist()}."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Error creating pandas pivot table: {e}") from e

    def polars(
        self,
        index: List[str],
        columns: List[str],
        filter: Optional[Dict[str, Any]] = None,
        index_formatter: Optional[Callable] = None,
        column_formatter: Optional[Callable] = None,
        session: Union[int, str, None] = None,
    ):
        """Returns data as a Polars DataFrame.

        Args:
            session: Optional session filter (int, "latest", or None).
        """
        # TODO: Implement polars export (potentially more efficient pivot)
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "polars is required for .polars() export. Please install it."
            )

        print(
            f"[Placeholder] Exporting to Polars: index={index}, columns={columns}, filter={filter}"
        )
        data = self.read(session=session, **(filter or {}))
        if not data:
            return pl.DataFrame()  # Return empty DataFrame if no data

        # Polars pivot requires slightly different handling
        # It's often better to have data in long format for Polars
        # This is a simplified pivot, real use might need more complex reshaping
        df = pl.from_dicts(data)
        try:
            # Ensure all index/column keys exist before pivoting
            missing_keys = [k for k in index + columns if k not in df.columns]
            if missing_keys:
                raise KeyError(
                    f"Cannot pivot: Key(s) {missing_keys} not found in data columns {df.columns}. Ensure index/columns keys exist in run_info, step, or metric definitions."
                )

            # Use "on" instead of deprecated "columns"
            pivot_df = df.pivot(values="value", index=index, on=columns)

            if column_formatter:
                print(
                    "[Warning] Polars column_formatter is not directly applied during pivot. Manual renaming might be needed."
                )
            if index_formatter:
                print(
                    "[Warning] Polars index_formatter is not directly applied during pivot. Manual renaming might be needed."
                )

            return pivot_df
        except Exception as e:
            raise RuntimeError(f"Error creating polars pivot table: {e}") from e

    def close(self):
        """Closes the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            # print("[Reader] DB connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
