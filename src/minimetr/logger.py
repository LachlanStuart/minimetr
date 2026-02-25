# Logger implementation will go here

import sqlite3
import threading
from queue import Queue, Empty
import time
import json
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union, List
import weakref
import atexit

# Type alias for the tuple representation of a dictionary, used for hashing/caching
DictTuple = Tuple[Tuple[str, Any], ...]

# Command types for the writer queue
# ('FLUSH', step_tuple, metrics_dict_for_that_step)
FlushCommand = Tuple[str, DictTuple, Dict[DictTuple, float]]
StopCommand = Tuple[str, None] # ('STOP', None)
QueueItem = Union[FlushCommand, StopCommand]

# Keep track of active loggers for atexit cleanup
_active_loggers: weakref.WeakValueDictionary[int, 'Logger'] = weakref.WeakValueDictionary()

def _cleanup_atexit():
    """Gracefully close any remaining active loggers on Python exit."""
    # Called on Python exit to attempt closing active loggers
    loggers_to_close = list(_active_loggers.values())
    if loggers_to_close:
        print(f"[minimetr atexit] Closing {len(loggers_to_close)} active logger(s)...")
        for logger in loggers_to_close:
            try:
                # Check if already closed to avoid redundant closing attempts
                if not logger._closed:
                    logger.close()
            except Exception as e:
                print(f"[minimetr atexit] Error closing logger for {logger._db_path}: {e}")

# Register the cleanup function to be called at interpreter exit
atexit.register(_cleanup_atexit)

class Logger:
    """Logs metrics efficiently to a minimetr SQLite database using a background thread.

    Handles deduplication of step contexts and metric definitions, batches writes,
    and stores float values compactly as binary blobs.

    Attributes:
        db_path (str): Path to the SQLite database file.
        run_info (Dict[str, Any]): Metadata associated with this logging session.
        auto_flush_on_new_step (bool): If True (default), automatically flushes the
            previous step's buffer when a new step context is encountered.
    """

    # Schema definition
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS Sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_info_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS Steps (
        step_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        step_json TEXT NOT NULL,
        UNIQUE(session_id, step_json),
        FOREIGN KEY (session_id) REFERENCES Sessions (session_id)
    );
    CREATE TABLE IF NOT EXISTS MetricDefinitions (
        metric_def_id INTEGER PRIMARY KEY AUTOINCREMENT,
        definition_json TEXT UNIQUE NOT NULL
    );
    CREATE TABLE IF NOT EXISTS MetricDefinitionSets (
        definition_set_id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_def_ids_json TEXT UNIQUE NOT NULL -- JSON array of metric_def_id
    );
    CREATE TABLE IF NOT EXISTS DataPoints (
        data_point_id INTEGER PRIMARY KEY AUTOINCREMENT,
        step_id INTEGER NOT NULL,
        definition_set_id INTEGER NOT NULL,
        values_blob BLOB NOT NULL,
        FOREIGN KEY (step_id) REFERENCES Steps (step_id),
        FOREIGN KEY (definition_set_id) REFERENCES MetricDefinitionSets (definition_set_id)
    );
    """

    def __init__(self,
                 db_path: str,
                 run_info: Optional[Dict[str, Any]] = None,
                 queue_timeout: float = 1.0,
                 auto_flush_on_new_step: bool = True):
        """Initializes the logger, database, session, and background writer thread.

        Args:
            db_path: Path to the SQLite database file. Will be created if it doesn't exist.
            run_info: A dictionary containing metadata for this session (e.g., hyperparameters,
                      model name). Keys must not conflict with keys used in `step_def`.
            queue_timeout: Timeout in seconds for the background writer waiting for new items.
            auto_flush_on_new_step: If True, automatically queue the previous step's data for
                                       writing when `log` or `new_step` is called with a
                                       different step context.
        """
        self._db_path: str = db_path
        self._run_info: Dict[str, Any] = run_info if run_info is not None else {}
        # Internal buffer: {step_tuple: {metric_tuple: value}}
        self._buffer: Dict[DictTuple, Dict[DictTuple, float]] = {}
        self._write_queue: Queue[QueueItem] = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._active_step_tuple: Optional[DictTuple] = None # The step context currently being logged to
        self._queue_timeout: float = queue_timeout
        self._auto_flush_on_new_step: bool = auto_flush_on_new_step
        self._session_id: Optional[int] = None # Set in _init_db_and_session
        self._lock: threading.Lock = threading.Lock() # Protects _buffer and _active_step_tuple
        self._closed: bool = False

        # Internal caches for IDs (populated by worker)
        # These are shared with the worker thread but primarily written by it.
        # Read access from main thread is minimal/not performance critical.
        self._metric_def_cache: Dict[DictTuple, int] = {}
        self._metric_set_cache: Dict[Tuple[int, ...], int] = {}

        try:
            self._init_db_and_session()
            self._start_worker()
            _active_loggers[id(self)] = self # Register for atexit cleanup
        except Exception:
            self._closed = True # Ensure logger is marked closed on init failure
            raise

    def _init_db_and_session(self):
        """(Internal) Connects to the DB, creates schema, creates a session entry."""
        try:
            # Use a temporary connection for setup
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executescript(self._SCHEMA)

            # Create Session
            run_info_json = json.dumps(self._run_info, sort_keys=True)
            cursor = conn.cursor()
            # Check if exact run_info already exists (optional, allows reusing session ID)
            # cursor.execute("SELECT session_id FROM Sessions WHERE run_info_json = ?", (run_info_json,))
            # existing = cursor.fetchone()
            # if existing:
            #     self._session_id = existing[0]
            # else:
            cursor.execute(
                "INSERT INTO Sessions (run_info_json) VALUES (?) RETURNING session_id",
                (run_info_json,),
            )
            self._session_id = cursor.fetchone()[0]
            conn.commit()
            # print(f"[Main] Initialized DB and created session_id: {self._session_id}")
            conn.close()
        except sqlite3.Error as e:
            print(f"Database error during initialization: {e}")
            self._closed = True  # Prevent further operations
            raise

    def _start_worker(self):
        """(Internal) Starts the background writer thread."""
        if self._closed:
            return
        self._worker_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._worker_thread.start()

    def _dict_to_tuple(self, d: Dict[str, Any]) -> DictTuple:
        """(Internal) Converts dict to a sortable, hashable tuple representation."""
        return tuple(sorted(d.items()))

    def _get_or_insert_step_id(self, conn: sqlite3.Connection, session_id: int, step_tuple: DictTuple) -> int:
        """(Worker) Gets step_id for a session/step_tuple, inserts if needed."""
        cursor = conn.cursor()
        step_json = json.dumps(list(step_tuple), sort_keys=True)
        cursor.execute(
            "SELECT step_id FROM Steps WHERE session_id = ? AND step_json = ?",
            (session_id, step_json),
        )
        row = cursor.fetchone()
        if row:
            return row[0]
        else:
            cursor.execute(
                "INSERT INTO Steps (session_id, step_json) VALUES (?, ?) RETURNING step_id",
                (session_id, step_json),
            )
            step_id = cursor.fetchone()[0]
            # No need to commit here, commit happens after DataPoints insert
            return step_id

    def _get_or_insert_metric_def_id(self, conn: sqlite3.Connection, metric_def_tuple: DictTuple) -> int:
        """(Worker) Gets metric_def_id, using cache, inserts if needed."""
        if metric_def_tuple in self._metric_def_cache:
            return self._metric_def_cache[metric_def_tuple]

        cursor = conn.cursor()
        metric_def_json = json.dumps(list(metric_def_tuple), sort_keys=True)
        cursor.execute(
            "SELECT metric_def_id FROM MetricDefinitions WHERE definition_json = ?",
            (metric_def_json,),
        )
        row = cursor.fetchone()
        if row:
            id_ = row[0]
            self._metric_def_cache[metric_def_tuple] = id_
            return id_
        else:
            cursor.execute(
                "INSERT INTO MetricDefinitions (definition_json) VALUES (?) RETURNING metric_def_id",
                (metric_def_json,),
            )
            id_ = cursor.fetchone()[0]
            self._metric_def_cache[metric_def_tuple] = id_
            # No need to commit here, commit happens after DataPoints insert
            return id_

    def _get_or_insert_metric_set_id(self, conn: sqlite3.Connection, metric_def_ids: Tuple[int, ...]) -> int:
        """(Worker) Gets definition_set_id, using cache, inserts if needed."""
        if metric_def_ids in self._metric_set_cache:
            return self._metric_set_cache[metric_def_ids]

        cursor = conn.cursor()
        metric_def_ids_json = json.dumps(list(metric_def_ids))
        cursor.execute(
            "SELECT definition_set_id FROM MetricDefinitionSets WHERE metric_def_ids_json = ?",
            (metric_def_ids_json,),
        )
        row = cursor.fetchone()
        if row:
            id_ = row[0]
            self._metric_set_cache[metric_def_ids] = id_
            return id_
        else:
            cursor.execute(
                "INSERT INTO MetricDefinitionSets (metric_def_ids_json) VALUES (?) RETURNING definition_set_id",
                (metric_def_ids_json,),
            )
            id_ = cursor.fetchone()[0]
            self._metric_set_cache[metric_def_ids] = id_
            # No need to commit here, commit happens after DataPoints insert
            return id_

    def _process_log_command(self, conn: sqlite3.Connection, session_id: int, step_tuple: DictTuple, metrics_dict: Dict[DictTuple, float]):
        """(Worker) Processes a batch of metrics for a single step, writing to DB."""
        if not metrics_dict:  # Prevent writing empty DataPoints rows
            return

        # print(f"[Worker] Processing step: {step_tuple}, {len(metrics_dict)} metrics")
        cursor = conn.cursor()
        try:
            # 1. Get Step ID
            step_id = self._get_or_insert_step_id(conn, session_id, step_tuple)

            # 2. Get Metric Definition IDs & Values (ordered)
            metric_def_ids = []
            metric_values = []
            sorted_metric_items = sorted(
                metrics_dict.items()
            )  # Sort by metric_def_tuple

            for metric_def_tuple, value in sorted_metric_items:
                metric_def_id = self._get_or_insert_metric_def_id(
                    conn, metric_def_tuple
                )
                metric_def_ids.append(metric_def_id)
                metric_values.append(value)

            # 3. Get Metric Definition Set ID
            metric_def_ids_tuple = tuple(metric_def_ids)
            metric_set_id = self._get_or_insert_metric_set_id(
                conn, metric_def_ids_tuple
            )

            # 4. Prepare and Insert Data Point
            values_array = np.array(metric_values, dtype=np.float32)
            values_blob = values_array.tobytes()

            cursor.execute(
                "INSERT INTO DataPoints (step_id, definition_set_id, values_blob) VALUES (?, ?, ?)",
                (step_id, metric_set_id, values_blob),
            )
            conn.commit()  # Commit transaction for this step
            # print(f"[Worker] Inserted datapoint for step_id {step_id}")
        except sqlite3.Error as e:
            print(f"[Worker] Database error processing step {step_tuple}: {e}")
            conn.rollback()  # Rollback on error for this step

    def _writer_loop(self):
        """(Internal) Background thread loop to process the write queue."""
        if self._session_id is None:
            return  # Should not happen if init successful

        conn = sqlite3.connect(
            self._db_path, check_same_thread=False, timeout=10.0
        )  # Longer timeout
        conn.execute("PRAGMA journal_mode=WAL;")
        # print("[Worker] Started")

        while True:
            item = None  # Ensure item is defined for finally block
            try:
                item = self._write_queue.get(timeout=self._queue_timeout)
                if item is None:  # Allow None to be queued explicitly
                    # This case might happen if None is put directly (shouldn't normally)
                    # or if queue is emptied before STOP is processed? Unlikely.
                    print("[Worker] Warning: Received None item unexpectedly.")
                    continue  # Skip processing, but call task_done in finally

                # Check item type and process
                if isinstance(item, tuple):
                    command = item[0]
                    payload = item[1]

                    if command == "STOP":
                        # print("[Worker] STOP command received.")
                        break  # Exit loop -> finally will call task_done
                    elif command == "FLUSH":
                        step_tuple = payload[0]
                        metrics_to_flush = payload[1]
                        if metrics_to_flush:  # Only process if there's actually data
                            self._process_log_command(
                                conn, self._session_id, step_tuple, metrics_to_flush
                            )
                        # else:
                        # print(f"[Worker] Skipping empty flush for step: {step_tuple}")
                    else:
                        # Should not happen with current QueueItem types
                        print(f"[Worker] Unknown command type: {command}")
                else:
                    print(f"[Worker] Received invalid item type in queue: {type(item)}")

            except Empty:
                # Queue was empty, loop continues, no task to mark done
                continue
            except Exception as e:
                # Log unexpected errors and continue (or maybe break?)
                print(f"[Worker] Unexpected error processing item {item}: {e}")
                import traceback

                traceback.print_exc()
                # We still need task_done to be called in finally
                # Optionally add a delay or different handling here
            finally:
                # Crucially, mark the task as done even if processing failed
                if item is not None:
                    self._write_queue.task_done()

        # Loop exited (STOP received)
        # Ensure the STOP command itself is marked as done
        if item is not None and isinstance(item, tuple) and item[0] == "STOP":
            # Check if task_done was already called in the finally block above
            # It should have been, but adding a check for robustness if needed.
            # self._write_queue.task_done() # Already done in finally
            pass

        conn.close()
        # print("[Worker] Stopped")

    def log(self, step_def: Dict[str, Any], value: float, **metric_def):
        """Logs a single metric value for a given step context.

        Checks for conflicting keys/values between the session's `run_info`,
        the provided `step_def`, and the `metric_def`.

        If `auto_flush_on_new_step` is True, calling this with a different `step_def`
        """Logs a single metric value for a given step context."""
        if self._closed:
            raise RuntimeError("Logger is closed.")

        # --- Key Collision Check ---
        # Check for conflicting values if the same key exists in multiple scopes
        key_sources = [self._run_info, step_def, metric_def]
        all_keys = (
            set(self._run_info.keys()) | set(step_def.keys()) | set(metric_def.keys())
        )
        merged_context: Dict[str, Any] = {}
        for key in all_keys:
            values_found = []
            sources_found = []
            if key in self._run_info:
                values_found.append(self._run_info[key])
                sources_found.append("run_info")
            if key in step_def:
                values_found.append(step_def[key])
                sources_found.append("step_def")
            if key in metric_def:
                values_found.append(metric_def[key])
                sources_found.append("metric_def")

            # Check for conflicts (more than one unique non-None value)
            unique_values = set(
                v for v in values_found if v is not None
            )  # Simple non-None comparison
            if len(unique_values) > 1:
                raise ValueError(
                    f"Conflicting values for key '{key}' found in scopes {sources_found}: {values_found}"
                )

            # Use the first value found (they are guaranteed to be the same if multiple)
            if values_found:
                merged_context[key] = values_found[0]
        # --- End Key Collision Check ---

        # Note: We buffer using the original step_def, not the merged_context
        # The merging happens again in the Reader based on stored definitions
        step_tuple = self._dict_to_tuple(step_def)
        metric_tuple = self._dict_to_tuple(metric_def)

        with self._lock:
            # Auto-flush if step changed and flag is enabled
            if (
                self._auto_flush_on_new_step
                and self._active_step_tuple is not None
                and step_tuple != self._active_step_tuple
            ):
                self._flush_step_internal(self._active_step_tuple)

            # Add metric to buffer for the current step_tuple
            if step_tuple not in self._buffer:
                self._buffer[step_tuple] = {}
            # Overwrite duplicate metric for the *same* step_tuple silently
            self._buffer[step_tuple][metric_tuple] = value
            self._active_step_tuple = step_tuple

    def new_step(self, **step_def):
        """Creates a Step object for logging metrics associated with this step context."""
        if self._closed:
            raise RuntimeError("Logger is closed.")

        # --- Key Collision Check (Run vs Step only) ---
        # We only check run_info vs step_def here, as metric_def isn't known yet.
        # The full check happens in log().
        colliding_keys = set(self._run_info.keys()) & set(step_def.keys())
        for key in colliding_keys:
            if self._run_info[key] != step_def[key]:
                raise ValueError(
                    f"Conflicting values for key '{key}' found in run_info and step_def: {self._run_info[key]} != {step_def[key]}"
                )
        # --- End Key Collision Check ---

        step_tuple = self._dict_to_tuple(step_def)

        with self._lock:
            # Auto-flush if step changed and flag is enabled
            if (
                self._auto_flush_on_new_step
                and self._active_step_tuple is not None
                and step_tuple != self._active_step_tuple
            ):
                self._flush_step_internal(self._active_step_tuple)

            # Ensure buffer exists even if no metrics logged yet via .log
            if step_tuple not in self._buffer:
                self._buffer[step_tuple] = {}
            self._active_step_tuple = step_tuple  # Set the new active step

        # Pass the step_def dict itself for potential later flushing by Step.__del__
        return Step(self, step_def)

    def _flush_step_internal(self, step_tuple: DictTuple):
        """Internal method to flush a specific step tuple. Assumes lock is held."""
        if step_tuple in self._buffer:
            metrics_to_flush = self._buffer.pop(step_tuple)
            if metrics_to_flush:  # Only queue if buffer wasn't empty
                # Put ('FLUSH', tuple, metrics_dict) command
                self._write_queue.put(("FLUSH", step_tuple, metrics_to_flush))
            # else: Buffer existed but was empty, do nothing
        # else: step_tuple not in buffer (already flushed or never used)

    def flush(self, step_def: Optional[Dict[str, Any]] = None):
        """Flushes buffered data for a specific step context or the active one."""
        if self._closed:
            raise RuntimeError("Logger is closed.")

        with self._lock:
            if step_def is not None:
                target_tuple = self._dict_to_tuple(step_def)
            else:
                target_tuple = self._active_step_tuple

            if target_tuple:
                self._flush_step_internal(target_tuple)
                # Clear active step if it was the one explicitly flushed
                if target_tuple == self._active_step_tuple:
                    self._active_step_tuple = None

    def close(self):
        """Flushes all remaining buffers and shuts down the writer thread gracefully."""
        if self._closed:
            return
        self._closed = True  # Prevent new logs/steps

        with self._lock:
            # Flush any remaining active/inactive buffers
            if self._active_step_tuple and self._active_step_tuple in self._buffer:
                self._flush_step_internal(self._active_step_tuple)
            remaining_step_tuples = list(self._buffer.keys())
            for step_tuple in remaining_step_tuples:
                self._flush_step_internal(step_tuple)
            self._buffer.clear()

        # Signal worker to stop and wait for queue to empty and thread to finish
        if self._worker_thread:
            self._write_queue.put(("STOP", None))
            self._write_queue.join()  # Wait for all tasks to be marked done
            self._worker_thread.join()  # Wait for thread to terminate

        # Unregister from atexit
        if id(self) in _active_loggers:
            del _active_loggers[id(self)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Step:
    """Represents a specific step context for logging."""

    def __init__(self, logger: Logger, step_def: Dict[str, Any]):
        # Use weakref to avoid circular reference preventing logger GC if Step obj persists
        self._logger_ref = weakref.ref(logger)
        self._step_def = step_def
        # Keep tuple for potential use in __del__ but primarily pass dict to flush
        self._step_def_tuple = logger._dict_to_tuple(step_def)
        self._dirty = False

    def log(self, value: float, **metric_def):
        """Logs a metric for this specific step context."""
        logger = self._logger_ref()
        if logger and not logger._closed:
            self._dirty = True
            logger.log(self._step_def, value, **metric_def)
        elif not logger:
            print("[Warning][Step] Logger already garbage collected. Cannot log.")
        # If logger is closed, log does nothing silently

    def flush(self):
        """Flushes data associated with this specific step context."""
        logger = self._logger_ref()
        if logger and not logger._closed:
            self._dirty = False
            logger.flush(self._step_def)
        elif not logger:
            print("[Warning][Step] Logger already garbage collected. Cannot flush.")

    def __del__(self):
        """Attempt to flush remaining data on garbage collection."""
        # __del__ is unreliable; using atexit in Logger is preferred for shutdown.
        # This is a best-effort attempt if a Step object is GC'd while logger is active.
        logger = self._logger_ref()
        if self._dirty and logger and not logger._closed:
            # Check buffer directly via lock (might be slow in __del__)
            with logger._lock:
                needs_flush = (
                    self._step_def_tuple in logger._buffer
                    and logger._buffer[self._step_def_tuple]
                )
            if needs_flush:
                try:
                    # Use the dict version for the public flush method
                    logger.flush(self._step_def)
                except Exception as e:
                    # Avoid exceptions escaping __del__
                    print(f"[GC][Step] Error flushing step {self._step_def} on GC: {e}")
