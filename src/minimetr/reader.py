# Reader implementation will go here

import sqlite3
import json
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Tuple, Set


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
    def keys(self) -> Dict[str, List[str]]:
        """Returns a dictionary categorizing all unique keys found across sessions, steps, and metrics."""
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

        for step_def in self._step_cache.values():
            keys_by_type["step"].update(step_def.keys())
        # Remove any keys that might have been wrongly assigned if present in run_info
        # keys_by_type['step'] -= keys_by_type['run']

        for metric_def in self._metric_def_cache.values():
            keys_by_type["metric"].update(metric_def.keys())

        self._keys_cache = {k: sorted(list(v)) for k, v in keys_by_type.items()}
        return self._keys_cache

    def read(self, **filters) -> List[Dict[str, Any]]:
        """Reads data points, applying filters across all key types."""
        self._load_definitions()  # Ensure definition caches are populated
        conn = self._get_db_connection()
        cursor = conn.cursor()

        # TODO: Optimize filtering - currently loads all then filters in Python
        # A more performant approach would build dynamic SQL WHERE clauses
        # based on indexed JSON fields or dedicated columns if performance critical.
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
                print(
                    f"[Warning] Skipping data point due to missing definition: step_id={step_id}, session_id={session_id}, def_set_id={def_set_id}"
                )
                continue

            # Decode values blob
            values = np.frombuffer(values_blob, dtype=np.float32)
            if len(values) != len(metric_def_ids):
                print(
                    f"[Warning] Skipping data point due to value/def mismatch: step_id={step_id}"
                )
                continue

            # Combine run_info, step context, metric def, and value
            for i, metric_def_id in enumerate(metric_def_ids):
                metric_def = self._metric_def_cache.get(metric_def_id)
                if metric_def is None:
                    print(
                        f"[Warning] Skipping metric due to missing metric definition: metric_def_id={metric_def_id}"
                    )
                    continue

                # Create the full record dictionary
                # Order: value, run_info, step_context, metric_def
                record = {"value": values[i], **run_info, **step_context, **metric_def}

                # Apply static filters first (cheaper)
                match = True
                for key, filter_val in static_filters.items():
                    # Check if key exists in the combined record
                    if key not in record or record[key] != filter_val:
                        match = False
                        break
                if not match:
                    continue

                # Apply lambda filters
                for key, filter_func in lambda_filters.items():
                    # Check if key exists before applying lambda
                    if key not in record or not filter_func(record[key]):
                        match = False
                        break
                if not match:
                    continue

                # If all filters passed
                results.append(record)

        return results

    def pivot(
        self,
        index: List[str],
        columns: List[str],
        filter: Optional[Dict[str, Any]] = None,
    ):
        """Pivots the data."""
        # TODO: Implement pivot logic using pandas or numpy
        print(
            f"[Placeholder] Pivoting data: index={index}, columns={columns}, filter={filter}"
        )
        raise NotImplementedError("Pivot functionality not yet implemented.")

    def pandas(
        self,
        index: List[str],
        columns: List[str],
        filter: Optional[Dict[str, Any]] = None,
        index_formatter: Optional[Callable] = None,
        column_formatter: Optional[Callable] = None,
    ):
        """Returns data as a Pandas DataFrame."""
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
        data = self.read(**(filter or {}))
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
    ):
        """Returns data as a Polars DataFrame."""
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
        data = self.read(**(filter or {}))
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

            pivot_df = df.pivot(values="value", index=index, columns=columns)

            # Polars doesn't easily support multi-index formatters like pandas
            # Users might need to handle renaming columns post-pivot if needed
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
