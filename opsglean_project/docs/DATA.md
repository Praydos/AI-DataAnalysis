# Data Source

OpsGlean uses the **HDFS_v1** log dataset from [LogHub](https://github.com/logpai/loghub),
a widely-used benchmark in log-based anomaly detection research (used in papers such as
DeepLog and LogAnomaly). Real production logs from a Hadoop Distributed File System,
labeled at the block level as `Normal` or `Anomaly`.

Data is **not committed to this repo** (see `.gitignore`) due to file size. To reproduce,
download the HDFS_v1 package from LogHub and place the files below into `data/raw/`.

## Files

| File | Rows | Description |
|---|---|---|
| `HDFS.log` | ~11.1M lines | Raw log file. One line per log event. **Not used directly** — pre-processed into the files below. |
| `anomaly_label.csv` | 575,061 | Ground truth: `BlockId, Label` (`Normal` / `Anomaly`) |
| `HDFS_log_templates.csv` | 29 | The 29 distinct log event templates (E1–E29) mined from the raw log via the Drain parsing algorithm |
| `Event_occurrence_matrix.csv` | 575,061 | One row per block: count of each event type (E1...E29) that occurred for that block. **Used for the classical ML baseline.** |
| `Event_traces.csv` | 575,061 | One row per block: full ordered event sequence (e.g. `[E5,E22,E5,...]`), plus time-interval and latency features. **Used for sequence models (e.g. DeepLog-style LSTM).** |
| `HDFS.npz` | 575,061 | NumPy archive with `x_data` (event sequences, array of lists) and `y_data` (binary labels, 0=normal/1=anomaly). Same content as `Event_traces.csv` in a directly-loadable format. |

## Key facts (established during EDA)

- **Class imbalance**: 2.93% of blocks are anomalous (16,838 / 575,061). Accuracy is not
  a meaningful metric here — use precision, recall, F1, and PR-AUC instead.
- **Sequence length**: median 19 events per block, 99th percentile = 33, max = 298.
- All train/test splits must be **stratified** on the label to preserve this ratio.

## Regenerating processed data

Processed data (train/test splits) is not committed either — it's produced by scripts in
`src/` from the raw files above. See `src/prepare_baseline_data.py`.
