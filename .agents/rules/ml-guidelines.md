# Machine Learning & Data Protection Guidelines

This rule enforces data privacy, context conservation, and storage limits during machine learning and dataset workflows.

## 1. Raw Dataset Dump Prohibition
To prevent catastrophic context explosion and preserve token bandwidth:
- **Never print raw dataset dumps into context or terminal outputs**:
  - Do NOT emit raw table contents, raw file streams, or unsummarized lines from tabular, columnar, or serialized datasets (including `.csv`, `.tsv`, `.parquet`, `.feather`, `.arrow`, `.jsonl`, `.h5`, `.npy`, `.npz`).
- **Strict Inspection Restrictions**:
  - Restrict dataset inspection strictly to **`.head(5)`** (or `.sample(5)`) for previewing rows.
  - Display structural schemas only: column names, data types (`dtypes`), missing value counts (`isna().sum()`), and key summary statistics (`describe()` or histogram summaries).
  - For large text or sequence corpora, truncate sample string lengths to a maximum of 200 characters per row.

## 2. Model Checkpoint Storage Restrictions
To avoid disk exhaustion during iterative training and fine-tuning runs:
- **Rolling Checkpoint Limit**:
  - Restrict checkpoint saving strictly to a rolling maximum of 3 checkpoints (`save_total_limit=3` in Hugging Face `TrainingArguments`, PyTorch Lightning `ModelCheckpoint(save_top_k=3)`, or custom training loops).
- **Cleanup of Stale Checkpoints**:
  - Delete temporary interim checkpoints once a training run concludes or when a final serialized model bundle is exported.
- **Disk Space Pre-Check**:
  - Verify available disk volume before initiating full checkpoint writing for multi-gigabyte foundation or diffusion models.
