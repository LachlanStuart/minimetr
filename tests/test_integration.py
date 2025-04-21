# Placeholder for integration tests

import pytest
import os
import tempfile
import minimetr
import time
import numpy as np
from typing import List, Dict, Any
import gc  # For testing __del__


# Helper function to compare lists of dictionaries ignoring order and float precision
def assert_records_equal(
    list1: List[Dict[str, Any]], list2: List[Dict[str, Any]], message=""
):
    assert len(list1) == len(list2), (
        f"Length mismatch: {len(list1)} != {len(list2)}. {message}"
    )

    # Convert to comparable format (tuple of sorted items, convert np.float32 to float, round floats)
    def dict_to_comparable(d):
        items = []
        for k, v in sorted(d.items()):
            if isinstance(v, (float, np.float32)):
                py_float = float(v)
                items.append((k, round(py_float, 5)))
            elif isinstance(v, (list, tuple)):
                # Handle potential nested lists/tuples if necessary, convert to tuple
                try:
                    items.append((k, tuple(v)))
                except TypeError:
                    items.append((k, v))  # If not convertible (e.g., list of dicts)
            else:
                items.append((k, v))
        return tuple(items)

    set1 = set(dict_to_comparable(d) for d in list1)
    set2 = set(dict_to_comparable(d) for d in list2)

    assert set1 == set2, (
        f"Set comparison failed. {message}\nSet 1: {set1}\nSet 2: {set2}"
    )


@pytest.fixture
def db_path():
    """Create a temporary database file path for testing."""
    # Use a unique filename in the current directory for easier inspection if needed
    # path = f"./test_minimetr_{int(time.time() * 1000)}.db"
    # yield path
    # if os.path.exists(path):
    #     os.remove(path)
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    print(f"Created temp db at: {path}")
    yield path
    if os.path.exists(path):
        os.remove(path)
        print(f"Removed temp db: {path}")


def test_log_and_read_auto_flush(db_path):
    """Tests logging with auto-flush enabled (default) and reading back."""
    run_info = {"model": "auto_flush_model", "run_id": "auto_flush_run", "lr": 0.001}
    _perform_log_and_read_test(db_path, run_info, auto_flush=True)


def test_log_and_read_manual_flush(db_path):
    """Tests logging with auto-flush disabled, requiring explicit flushes."""
    run_info = {
        "model": "manual_flush_model",
        "run_id": "manual_flush_run",
        "lr": 0.002,
    }
    _perform_log_and_read_test(db_path, run_info, auto_flush=False)


def _perform_log_and_read_test(db_path, run_info, auto_flush):
    """Core logic for logging and reading tests."""
    print(f"\n--- Testing with auto_flush={auto_flush}, run_info={run_info} ---")
    # Logging Phase
    with minimetr.Logger(
        db_path, run_info=run_info, auto_flush_on_new_step=auto_flush
    ) as logger:
        step_1_train_def = {"step_num": 1, "epoch": 0, "phase": "train"}
        step_1_val_def = {"step_num": 1, "epoch": 0, "phase": "val"}
        step_2_train_def = {"step_num": 2, "epoch": 0, "phase": "train"}

        # --- Log Step 1 Train ---
        step1_train_obj = logger.new_step(**step_1_train_def)
        step1_train_obj.log(0.1, metric="loss")
        step1_train_obj.log(0.9, metric="accuracy")
        if not auto_flush:
            step1_train_obj.flush()  # Manual flush needed

        # --- Log Step 1 Val ---
        # Auto-flush should trigger flush of step 1 train here if enabled
        step1_val_obj = logger.new_step(**step_1_val_def)
        step1_val_obj.log(0.2, metric="loss")
        if not auto_flush:
            step1_val_obj.flush()

        # --- Log Step 2 Train ---
        # Auto-flush should trigger flush of step 1 val here if enabled
        step2_train_obj = logger.new_step(**step_2_train_def)
        step2_train_obj.log(1.5, metric="loss")
        step2_train_obj.log(10.0, layer=0, metric="norm")

        # --- Test Resumable Step (Log more to Step 1 Train) ---
        print("Logging additional metric to step 1 train after potential flush...")
        step1_train_obj.log(50.0, metric="grad_norm")  # Log after step 2 started
        if not auto_flush:
            step1_train_obj.flush()  # Manual flush needed for the new metric
        # If auto_flush=True, this ^ metric won't be flushed until logger.close or next new_step

        # Need explicit flush for step 2 train if auto_flush=False
        if not auto_flush:
            step2_train_obj.flush()

        # Test Step GC Flush (only reliable in CPython, may not trigger immediately)
        # print("Testing Step GC flush...")
        # step_to_gc = logger.new_step(step_num=99, phase="gc_test")
        # step_to_gc.log(99.9, metric="gc_metric")
        # step_to_gc_def = step_to_gc._step_def # Keep def for reading check
        # del step_to_gc
        # gc.collect() # Try to force collection
        # time.sleep(0.2) # Give GC and writer thread time

        # logger.close() is called by context manager, ensures final flushes

    # Reading Phase
    with minimetr.Reader(db_path) as reader:
        # Expected data (incorporating run_info)
        expected_all = [
            # Step 1 Train (initial + resumable)
            {"value": 0.1, **run_info, **step_1_train_def, "metric": "loss"},
            {"value": 0.9, **run_info, **step_1_train_def, "metric": "accuracy"},
            {"value": 50.0, **run_info, **step_1_train_def, "metric": "grad_norm"},
            # Step 1 Val
            {"value": 0.2, **run_info, **step_1_val_def, "metric": "loss"},
            # Step 2 Train
            {"value": 1.5, **run_info, **step_2_train_def, "metric": "loss"},
            {
                "value": 10.0,
                **run_info,
                **step_2_train_def,
                "layer": 0,
                "metric": "norm",
            },
            # GC Test Step (if uncommented)
            # {'value': 99.9, **run_info, **step_to_gc_def, 'metric': 'gc_metric'}
        ]

        # Read all data
        print("Reading all data...")
        read_all = reader.read()
        assert_records_equal(read_all, expected_all, message="Read All Mismatch")

        # Test filtering still works
        print("Reading filtered by step context (phase=val)...")
        read_val = reader.read(phase="val")
        expected_val = [
            {"value": 0.2, **run_info, **step_1_val_def, "metric": "loss"},
        ]
        assert_records_equal(
            read_val, expected_val, message="Phase=Val Filter Mismatch"
        )

        print("Reading filtered by multiple keys (step_num=1, phase=train)...")
        read_s1_train = reader.read(step_num=1, phase="train")
        expected_s1_train = [
            {"value": 0.1, **run_info, **step_1_train_def, "metric": "loss"},
            {"value": 0.9, **run_info, **step_1_train_def, "metric": "accuracy"},
            {"value": 50.0, **run_info, **step_1_train_def, "metric": "grad_norm"},
        ]
        assert_records_equal(
            read_s1_train,
            expected_s1_train,
            message="Step=1, Phase=Train Filter Mismatch",
        )

        print(f"--- Test completed for auto_flush={auto_flush} ---")


# Keep the original test name structure if needed, but point to new functions
# def test_log_and_read(db_path):
#     test_log_and_read_auto_flush(db_path)
