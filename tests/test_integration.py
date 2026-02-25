# Placeholder for integration tests

import pytest
import os
import tempfile
import minimetr
import time
import numpy as np
from typing import List, Dict, Any
import gc  # For testing __del__
import pandas as pd
import polars as pl


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

# --- Logger Collision Tests ---


def test_logger_key_collision_error(db_path):
    """Tests that logger raises ValueError on conflicting key values."""
    run_info = {"model": "collision", "lr": 0.1}
    step_def = {"step": 1, "lr": 0.01}  # Conflicting lr
    metric_def = {"metric": "loss", "phase": "train"}
    step_def_ok = {"step": 1, "phase": "train"}  # Ok step
    metric_def_conflict = {"metric": "loss", "phase": "val"}  # Conflicting phase

    with minimetr.Logger(db_path, run_info=run_info) as logger:
        # Conflict between run_info and step_def
        with pytest.raises(ValueError, match=r"Conflicting values for key 'lr'.*"):
            logger.new_step(**step_def)

        # Need to create a valid step first
        step_ok = logger.new_step(**step_def_ok)

        # Conflict between step_def and metric_def
        with pytest.raises(ValueError, match=r"Conflicting values for key 'phase'.*"):
            step_ok.log(0.5, **metric_def_conflict)

        # Conflict between run_info and metric_def
        with pytest.raises(ValueError, match=r"Conflicting values for key 'lr'.*"):
            step_ok.log(0.6, metric="acc", lr=0.001)


def test_logger_key_collision_ok(db_path):
    """Tests that logger allows identical values for keys in different scopes."""
    run_info = {"model": "collision_ok", "lr": 0.1, "phase": "train"}
    step_def = {"step": 1, "lr": 0.1}  # Same lr as run_info
    metric_def = {"metric": "loss", "phase": "train"}  # Same phase as run_info

    with minimetr.Logger(db_path, run_info=run_info) as logger:
        step = logger.new_step(**step_def)
        # This should succeed as 'lr' and 'phase' match run_info
        step.log(0.5, **metric_def)
        # This should also succeed
        logger.log(step_def, 0.9, metric="acc", phase="train")
        logger.flush()
    # Further checks could read back the data if needed


# --- Reader Tests ---


@pytest.fixture(scope="module")
def populated_db_path():
    """Create and populate a db file once for all reader tests."""
    fd, path = tempfile.mkstemp(suffix="_reader.db")
    os.close(fd)
    print(f"\n[Fixture] Created populated db at: {path}")
    run_info1 = {"model": "reader_test", "run": 0, "lr": 0.1}
    run_info2 = {"model": "reader_test", "run": 1, "seed": 42}

    with minimetr.Logger(path, run_info=run_info1) as logger1:
        logger1.log({"step": 0, "phase": "train"}, 0.9, metric="loss", layer=0)
        logger1.log({"step": 0, "phase": "train"}, 0.8, metric="acc", layer=0)
        logger1.log({"step": 1, "phase": "train"}, 0.8, metric="loss", layer=0)
        logger1.log({"step": 1, "phase": "train"}, 0.85, metric="acc", layer=0)
        logger1.log({"step": 1, "phase": "val"}, 0.82, metric="loss", layer=0)
        logger1.log({"step": 1, "phase": "val"}, 0.83, metric="acc", layer=0)

    with minimetr.Logger(path, run_info=run_info2) as logger2:
        logger2.log({"step": 0, "phase": "train"}, 0.95, metric="loss", layer=0)
        # Log same metric again (should be overwritten in buffer before flush)
        logger2.log({"step": 0, "phase": "train"}, 0.96, metric="loss", layer=0)
        logger2.log({"step": 1, "phase": "train"}, 0.85, metric="loss", layer=1)

    yield path
    if os.path.exists(path):
        os.remove(path)
        print(f"[Fixture] Removed populated db: {path}")


def test_reader_keys(populated_db_path):
    """Tests the different key properties of the Reader."""
    with minimetr.Reader(populated_db_path) as reader:
        all_keys = reader.keys
        run_keys = reader.run_keys
        step_keys = reader.step_keys
        metric_keys = reader.metric_keys

        print(f"All Keys: {all_keys}")
        print(f"Run Keys: {run_keys}")
        print(f"Step Keys: {step_keys}")
        print(f"Metric Keys: {metric_keys}")

        expected_all = sorted(
            ["model", "run", "lr", "seed", "step", "phase", "metric", "layer"]
        )  # Value is implicit
        expected_run = sorted(["model", "run", "lr", "seed"])
        expected_step = sorted(["step", "phase"])
        expected_metric = sorted(["metric", "layer"])

        # Exclude implicit 'value' key for comparison with categorized keys
        assert all_keys == expected_all, "Mismatch in all_keys"
        assert run_keys == expected_run, "Mismatch in run_keys"
        assert step_keys == expected_step, "Mismatch in step_keys"
        assert metric_keys == expected_metric, "Mismatch in metric_keys"
        assert set(run_keys + step_keys + metric_keys) == set(all_keys)


def test_reader_pivot(populated_db_path):
    """Tests the pivot method."""
    with minimetr.Reader(populated_db_path) as reader:
        # Pivot loss by step and phase for run 0
        result = reader.pivot(
            index=["step"], columns=["phase"], filter={"run": 0, "metric": "loss"}
        )

        print(f"Pivot Result: {result}")

        expected_index = [(0,), (1,)]
        expected_cols = [("train",), ("val",)]
        expected_values = np.array([[0.9, np.nan], [0.8, 0.82]], dtype=np.float32)

        assert result.index_tuples == expected_index, "Pivot index mismatch"
        assert result.column_tuples == expected_cols, "Pivot columns mismatch"
        np.testing.assert_allclose(
            result.values_array, expected_values, rtol=1e-6, equal_nan=True
        )


def test_reader_pandas(populated_db_path):
    """Tests the pandas export method."""
    with minimetr.Reader(populated_db_path) as reader:
        # Get loss for run 0 as pandas df
        df = reader.pandas(
            index=["step"], columns=["phase"], filter={"run": 0, "metric": "loss"}
        )

        print(f"Pandas DataFrame:\n{df}")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert list(df.index) == [0, 1]
        assert list(df.columns) == ["train", "val"]
        # Check with index name specified
        expected_train_series = pd.Series(
            [0.9, 0.8], index=pd.Index([0, 1], name="step"), name="train"
        )
        expected_val_series = pd.Series(
            [np.nan, 0.82], index=pd.Index([0, 1], name="step"), name="val"
        )
        pd.testing.assert_series_equal(
            df["train"], expected_train_series, check_dtype=False, atol=1e-6
        )
        pd.testing.assert_series_equal(
            df["val"], expected_val_series, check_dtype=False, atol=1e-6
        )

        # Test with formatter
        df_fmt = reader.pandas(
            index=["step"],
            columns=["phase", "metric"],
            filter={"run": 0},
            column_formatter=lambda keys: f"{keys[0]}_{keys[1]}",
        )
        print(f"Formatted Pandas DataFrame:\n{df_fmt}")
        assert sorted(list(df_fmt.columns)) == sorted(
            ["train_loss", "train_acc", "val_loss", "val_acc"]
        )


def test_reader_polars(populated_db_path):
    """Tests the polars export method."""
    with minimetr.Reader(populated_db_path) as reader:
        # Get loss for run 0 as polars df
        df = reader.polars(
            index=["step"], columns=["phase"], filter={"run": 0, "metric": "loss"}
        )
        print(f"Polars DataFrame:\n{df}")
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (2, 3)  # Polars pivot includes index column(s)
        assert df.columns == ["step", "train", "val"]
        assert df["step"].to_list() == [0, 1]
        # Check values (handling potential nulls/NaNs) and float precision
        assert df["train"].to_list() == pytest.approx([0.9, 0.8], abs=1e-6)
        assert df["val"][0] is None
        assert df["val"][1] == pytest.approx(0.82, abs=1e-6)
