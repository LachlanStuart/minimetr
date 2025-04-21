# minimetr

A lightweight Python library for logging metrics from deep learning training runs to a compact SQLite database. Designed for efficient storage and analysis of potentially thousands of measurement events per run, focusing on event stream analysis rather than traditional time-series visualization like TensorBoard.

## Core Concepts

- **Compound Timestamps:** Each measurement point is associated with a flexible timestamp defined by a dictionary of key-value pairs (e.g., `{'step': 100, 'epoch': 1, 'phase': 'train'}`).
- **Compound Metric IDs:** Metrics are identified by a dictionary of key-value pairs (e.g., `{'layer': 3, 'metric': 'loss'}`), allowing for multi-dimensional slicing and dicing.
- **Sessions:** Each `Logger` instance represents a single session (e.g., a training run) and stores associated metadata (`run_info`) like model name, hyperparameters, etc.
- **Step Context:** Each measurement point is associated with a flexible step context defined by a dictionary of key-value pairs (e.g., `{'step_num': 100, 'epoch': 1, 'phase': 'train'}`). Each step context belongs to a session.
- **Metric Definitions:** Metrics are identified by a dictionary of key-value pairs (e.g., `{'layer': 3, 'metric': 'loss'}`), allowing for multi-dimensional slicing and dicing.
- **Compact Storage:**
    - Session metadata dictionaries (`run_info`) are stored once per session.
    - Step context dictionaries are deduplicated *within each session*.
    - Metric definition dictionaries are deduplicated globally.
    - The specific *ordered set* of metrics recorded together is deduplicated (`MetricDefinitionSet`).
    - The float32 measurement values corresponding to that ordered set are concatenated and stored as a single binary `BLOB`.
- **Buffered & Batched Writing:** Metrics for a specific step context are collected incrementally. A background thread handles writing data to the database. Flushing (writing buffered data) can be triggered explicitly or automatically when a new step context is encountered (configurable).
- **Resumable Steps:** Logging to a step context can be resumed even after it has been flushed. Subsequent logs will be written as new data points associated with the same step context when the next flush occurs.

## Database Schema

The data is stored in an SQLite database with the following schema:

**1. `Sessions` Table:** Stores unique run/session metadata.

| Column         | Type    | Description                                                |
| -------------- | ------- | ---------------------------------------------------------- |
| `session_id`   | INTEGER | PRIMARY KEY AUTOINCREMENT                                  |
| `run_info_json`| TEXT    | JSON string representation of the run_info dict.           |


**2. `Steps` Table:** Stores unique step context dictionaries, linked to a session.

| Column          | Type    | Description                                                |
| --------------- | ------- | ---------------------------------------------------------- |
| `step_id`       | INTEGER | PRIMARY KEY AUTOINCREMENT                                  |
| `session_id`    | INTEGER | FOREIGN KEY REFERENCES `Sessions(session_id)`              |
| `step_json`     | TEXT    | JSON string representation of the step context dict.       |
| *Constraint*    | UNIQUE  | `(session_id, step_json)`                                  |


**3. `MetricDefinitions` Table:** Stores unique metric identifier dictionaries.

| Column            | Type    | Description                                                      |
| ----------------- | ------- | ---------------------------------------------------------------- |
| `metric_def_id`   | INTEGER | PRIMARY KEY AUTOINCREMENT                                        |
| `definition_json` | TEXT    | UNIQUE. JSON string representation of the metric ID dictionary. |


**4. `MetricDefinitionSets` Table:** Stores the unique *ordered sequence* of metrics recorded together.

| Column              | Type    | Description                                                                      |
| ------------------- | ------- | -------------------------------------------------------------------------------- |
| `definition_set_id` | INTEGER | PRIMARY KEY AUTOINCREMENT                                                        |
| `metric_def_ids_json` | TEXT    | UNIQUE. JSON array of `metric_def_id`s, defining the order of values in the blob. |


**5. `DataPoints` Table:** Stores the actual measurement data blobs linked to steps and metric definition sets.

| Column          | Type    | Description                                                               |
| --------------- | ------- | ------------------------------------------------------------------------- |
| `data_point_id` | INTEGER | PRIMARY KEY AUTOINCREMENT                                                 |
| `step_id`       | INTEGER | FOREIGN KEY REFERENCES `Steps(step_id)`                                   |
| `definition_set_id`      | INTEGER | FOREIGN KEY REFERENCES `MetricDefinitionSets(definition_set_id)`         |
| `values_blob`   | BLOB    | Concatenated float32 values corresponding to `MetricDefinitionSets.metric_def_ids_json`. |


## Usage (Conceptual)

```python
# (Future API Example)
import minimetr
import time

# Connect to or create a database file, creating a new Session
logger = minimetr.Logger(
    "training_run.db",
    run_info={'model': 'resnet50', 'dataset': 'imagenet', 'lr': 0.01},
    auto_flush_on_new_step=True # Default is True
)

# Option 1: Log directly via logger

# Define the step context once
step_def = {"step_num": 100, "epoch": 1, "phase": "train"}

# Log metrics one by one for this step context
logger.log(step_def, 0.123, metric='loss')
logger.log(step_def, 0.95, metric='accuracy')
logger.log(step_def, 15.2, layer=0, param='weight', metric='norm')

# Logging with a *new* step context implicitly flushes the previous one (step_num 100, train)
val_step_def = {"step_num": 100, "epoch": 1, "phase": "validation"}
logger.log(val_step_def, 0.200, metric='loss')
logger.log(val_step_def, 0.91, metric='accuracy')

# Explicitly flush the last step context (optional, happens automatically on close)
logger.flush(val_step_def)

# Option 2: Use Step object

step_101_train = logger.new_step(step_num=101, epoch=1, phase="train")
# Methods on step_101_train implicitly flush val_step_def from Option 1

step_101_train.log(0.110, metric='loss')
step_101_train.log(0.96, metric='accuracy')
step_101_train.log(14.8, layer=0, param='weight', metric='norm')

# Creating a new Step object also flushes the previous one (if auto_flush_on_new_step=True)
step_101_val = logger.new_step(step_num=101, epoch=1, phase="validation")
step_101_val.log(0.195, metric='loss')
step_101_val.log(0.92, metric='accuracy')

# Flushing can still be done explicitly if needed
step_101_train.flush() # Flushes any remaining data for this specific step context
step_101_val.flush()

# The Step object will also attempt to flush upon garbage collection (CPython specific caveat)
# del step_101_val
# time.sleep(0.1) # Allow GC and background thread time

# Log more data to a previously flushed step
step_101_train.log(0.005, metric='weight_decay_loss')
step_101_train.flush() # Flush the newly added metric

logger.close() # Flushes all remaining buffered data for all steps in this session
```

## Reading Data (Conceptual)

Data reading focuses on retrieving points based on filtering criteria across *all* recorded keys (from the session's `run_info`, the step context, and the metric definitions) and provides convenient export formats.

```python
# (Future API Example)
import minimetr
import numpy as np
# Assume pandas and polars are installed if using those methods
# import pandas as pd
# import polars as pl

reader = minimetr.Reader("training_run.db")

# Inspecting Available Keys (Optional)

# Get all unique keys used across sessions (`run_info`), steps, and metrics
all_keys = reader.keys
# Example: {'run': ['model', 'dataset', 'lr'], 'step': ['step_num', 'epoch', 'phase'], 'metric': ['metric', 'layer', 'param']}

# Get all unique definitions recorded (as dicts)
# all_timestamps_dicts = reader.timestamps # List[dict] - Removed
# all_metrics_dicts = reader.metrics     # List[dict] - Removed


# Flexible Reading with Filtering
# The reader intelligently merges session `run_info`, step context, and metric keys
# for filtering and output.

# 1. Read all data points
# Returns: list[dict] where each dict starts with 'value' and contains merged keys
all_data = reader.read()
# Example: [
#   {'value': 0.123, 'model': 'resnet50', ..., 'step_num': 100, ..., 'metric': 'loss', ...},
#   {'value': 0.95, 'model': 'resnet50', ..., 'step_num': 100, ..., 'metric': 'accuracy', ...},
#   ...
# ]

# 2. Filter by any key(s)
validation_data = reader.read(phase='validation')

step_100_loss = reader.read(step_num=100, metric='loss')

resnet_layer0_norm = reader.read(model='resnet50', layer=0, metric='norm')

# 3. Filter using lambdas for numerical fields
early_steps_data = reader.read(step_num=lambda s: s < 50)

# high_accuracy = reader.read(phase='validation', metric='accuracy', value=lambda v: v > 0.9) # Value filter removed


# Pivot Table Functionality

# 1. Generic Pivot (returns namedtuple with index/column tuples and numpy array)
# PivotResult = namedtuple("PivotResult", ["index_tuples", "column_tuples", "values_array"])

pivot_result = reader.pivot(
    index=['epoch', 'step_num'],
    columns=['metric'],
    filter={'phase': 'validation', 'model': 'resnet50'}
)
# pivot_result.index_tuples: [(1, 100), (1, 200), ...]
# pivot_result.column_tuples: [('loss',), ('accuracy',)]
# pivot_result.values_array: np.array([[0.2, 0.91], [0.19, 0.92], ...]) # float32, NaNs for missing

# 2. Pandas DataFrame Export
pd_df = reader.pandas(
    index=['epoch', 'step_num'],
    columns=['metric', 'layer'],
    filter={'phase': 'train', 'step_num': lambda s: s % 10 == 0},
    column_formatter=lambda keys: "_".join(map(str, keys)) # Avoid MultiIndex columns
)
# Results in a pandas DataFrame with columns like 'loss_None', 'accuracy_None', 'norm_0', etc.

# 3. Polars DataFrame Export
pl_df = reader.polars(
    index=['epoch', 'step_num'], # Polars doesn't have a native multi-index like pandas
    columns=['metric'],      # Usually keep index/columns simpler or use formatters
    filter={'phase': 'validation'}
)

reader.close()
```

## Development Setup

This project uses `uv`
