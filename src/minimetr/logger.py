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

# Type alias for the tuple representation of a dictionary
DictTuple = Tuple[Tuple[str, Any], ...]

# Command types for the writer queue
LogCommand = Tuple[DictTuple, Dict[DictTuple, float]]  # step_tuple, metrics_dict
FlushCommand = Tuple[str, DictTuple]  # 'FLUSH', step_tuple
StopCommand = Tuple[str, None]  # 'STOP', None
QueueItem = Union[LogCommand, FlushCommand, StopCommand]

# Keep track of active loggers for atexit cleanup
_active_loggers = weakref.WeakValueDictionary()


def _cleanup_atexit():
    # Called on Python exit to attempt closing active loggers
    loggers_to_close = list(_active_loggers.values())
    if loggers_to_close:
        print(f"[minimetr atexit] Closing {len(loggers_to_close)} active logger(s)...")
        for logger in loggers_to_close:
            try:
                logger.close()
            except Exception as e:
                print(
                    f"[minimetr atexit] Error closing logger for {logger._db_path}: {e}"
                )


atexit.register(_cleanup_atexit)


class Logger:
    """Logs metrics to a minimetr SQLite database."""

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

    def __init__(
        self,
        db_path: str,
        run_info: Optional[Dict[str, Any]] = None,
        queue_timeout: float = 1.0,
        auto_flush_on_new_step: bool = True,
    ):
        """Initializes the logger, database, session, and background writer thread."""
        self._db_path = db_path
        self._run_info = run_info if run_info is not None else {}
        self._buffer: Dict[DictTuple, Dict[DictTuple, float]] = {}
        self._write_queue: Queue[Optional[QueueItem]] = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._active_step_tuple: Optional[DictTuple] = None
        self._queue_timeout = queue_timeout
        self._auto_flush_on_new_step = auto_flush_on_new_step
        self._session_id: Optional[int] = None
        self._lock = threading.Lock()  # Protect buffer access
        self._closed = False

        # Internal caches for IDs (populated by worker)
        # Step cache needs to be session-aware: Dict[session_id, Dict[step_tuple, step_id]]
        # For simplicity here, let worker handle lookups directly for now, no complex cache.
        # self._step_cache: Dict[DictTuple, int] = {}
        self._metric_def_cache: Dict[DictTuple, int] = {}
        self._metric_set_cache: Dict[Tuple[int, ...], int] = {}

        self._init_db_and_session()
        self._start_worker()
        _active_loggers[id(self)] = self  # Register for atexit cleanup

    def _init_db_and_session(self):
        """Connects to the database, creates tables, and creates a session entry."""
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
            print(f"[Main] Initialized DB and created session_id: {self._session_id}")
            conn.close()
        except sqlite3.Error as e:
            print(f"Database error during initialization: {e}")
            self._closed = True  # Prevent further operations
            raise

    def _start_worker(self):
        """Starts the background worker thread."""
        if self._closed:
            return
        self._worker_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._worker_thread.start()

    def _dict_to_tuple(self, d: Dict[str, Any]) -> DictTuple:
        """Converts a dictionary to a sorted tuple of items for consistent hashing/comparison."""
        return tuple(sorted(d.items()))

    def _get_or_insert_step_id(
        self, conn: sqlite3.Connection, session_id: int, step_tuple: DictTuple
    ) -> int:
        """Gets step_id for a given session and step_tuple, inserting if needed."""
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

    def _get_or_insert_metric_def_id(
        self, conn: sqlite3.Connection, metric_def_tuple: DictTuple
    ) -> int:
        """Gets metric_def_id, using cache, inserting if needed."""
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

    def _get_or_insert_metric_set_id(
        self, conn: sqlite3.Connection, metric_def_ids: Tuple[int, ...]
    ) -> int:
        """Gets definition_set_id, using cache, inserting if needed."""
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

    def _process_log_command(
        self,
        conn: sqlite3.Connection,
        session_id: int,
        step_tuple: DictTuple,
        metrics_dict: Dict[DictTuple, float],
    ):
        """Processes a batch of metrics for a single step."""
        if not metrics_dict:  # Prevent writing empty DataPoints rows
            return

        print(f"[Worker] Processing step: {step_tuple}, {len(metrics_dict)} metrics")
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
            print(f"[Worker] Inserted datapoint for step_id {step_id}")
        except sqlite3.Error as e:
            print(f"[Worker] Database error processing step {step_tuple}: {e}")
            conn.rollback()  # Rollback on error for this step

    def _writer_loop(self):
        """Background thread loop to process the write queue."""
        if self._session_id is None:
            return  # Should not happen if init successful

        conn = sqlite3.connect(
            self._db_path, check_same_thread=False, timeout=10.0
        )  # Longer timeout
        conn.execute("PRAGMA journal_mode=WAL;")
        print("[Worker] Started")

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
                        print("[Worker] STOP command received.")
                        break  # Exit loop -> finally will call task_done
                    elif command == "FLUSH":
                        step_tuple = payload
                        metrics_to_flush = item[
                            2
                        ]  # Get metrics passed with flush command
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
        print("[Worker] Stopped")

    def log(self, step_def: Dict[str, Any], value: float, **metric_def):
        """Logs a single metric value for a given step context."""
        if self._closed:
            raise RuntimeError("Logger is closed.")

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
            self._buffer[step_tuple][metric_tuple] = value
            self._active_step_tuple = step_tuple

    def new_step(self, **step_def):
        """Creates a Step object for logging metrics associated with this step context."""
        if self._closed:
            raise RuntimeError("Logger is closed.")

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
                print(f"[Main] Queued flush for step: {step_tuple}")
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
            else:
                print(
                    "[Main] Flush called with no active step and no specific step provided."
                )

    def close(self):
        """Flushes all remaining buffers and shuts down the writer thread gracefully."""
        if self._closed:
            return
        print("[Main] Logger closing...")
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
            print("[Main] Signaling worker thread to stop...")
            self._write_queue.put(("STOP", None))
            self._write_queue.join()  # Wait for all tasks to be marked done
            self._worker_thread.join()  # Wait for thread to terminate
            print("[Main] Worker thread joined.")

        # Unregister from atexit
        if id(self) in _active_loggers:
            del _active_loggers[id(self)]
        print("[Main] Logger closed.")

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

    def log(self, value: float, **metric_def):
        """Logs a metric for this specific step context."""
        logger = self._logger_ref()
        if logger and not logger._closed:
            logger.log(self._step_def, value, **metric_def)
        elif not logger:
            print("[Warning][Step] Logger already garbage collected. Cannot log.")
        # If logger is closed, log does nothing silently

    def flush(self):
        """Flushes data associated with this specific step context."""
        logger = self._logger_ref()
        if logger and not logger._closed:
            logger.flush(self._step_def)
        elif not logger:
            print("[Warning][Step] Logger already garbage collected. Cannot flush.")

    def __del__(self):
        """Attempt to flush remaining data on garbage collection."""
        # __del__ is unreliable; using atexit in Logger is preferred for shutdown.
        # This is a best-effort attempt if a Step object is GC'd while logger is active.
        logger = self._logger_ref()
        if logger and not logger._closed:
            # Check buffer directly via lock (might be slow in __del__)
            needs_flush = False
            with logger._lock:
                if (
                    self._step_def_tuple in logger._buffer
                    and logger._buffer[self._step_def_tuple]
                ):
                    needs_flush = True
            if needs_flush:
                print(
                    f"[GC][Step] Flushing step {self._step_def} on garbage collection."
                )
                try:
                    # Use the dict version for the public flush method
                    logger.flush(self._step_def)
                except Exception as e:
                    # Avoid exceptions escaping __del__
                    print(f"[GC][Step] Error flushing step {self._step_def} on GC: {e}")
