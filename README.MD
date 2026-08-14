# HBCD ECG Pipeline

A BIDS-style ECG quality-control and analysis workflow for task-based EEG/ECG recordings.

This repository contains a processing script, HBCD_ECG_pipeline_v6.py, that reads EEGLab .set files, extracts an ECG channel, crops the signal to the task window, performs signal cleaning and R-peak analysis, and generates plots and CSV summaries for QC review.

## What the pipeline does

The workflow is designed to:

- locate BIDS-style EEG/ECG recordings with matching sidecar files
- identify the ECG channel from channels.tsv, or fall back to a configured channel name
- crop the signal to the task window defined by events.tsv markers such as bas+ and TRSP
- filter the ECG signal and assess quality
- detect R-peaks and correct them using signal_fixpeaks
- flag abnormal RR intervals using the 20% change rule
- generate figures for raw ECG, filtered ECG, quality metrics, R-peaks, RR intervals, heart rate, HRV, and task isolation
- save summary CSV files for downstream review and analysis

## Input structure

The pipeline expects a BIDS-like directory structure such as:

```text
bids_dir/
  sub-<label>/
    ses-<label>/
      eeg/
        sub-<label>_ses-<label>_task-RS_acq-ecg_run-01_eeg.set
        sub-<label>_ses-<label>_task-RS_acq-ecg_run-01_channels.tsv
        sub-<label>_ses-<label>_task-RS_acq-ecg_run-01_events.tsv
```

The script looks for:

- EEGLab file: *_eeg.set
- channels sidecar: *_channels.tsv
- events sidecar: *_events.tsv

## Dependencies

The pipeline requires the following Python packages:

- numpy
- pandas
- matplotlib
- mne
- neurokit2
- scipy
- eeglabio

Install them in your Python environment before running the script.

## Quick start

Run the pipeline from the repository directory:

```bash
python HBCD_ECG_pipeline_v6.py <bids_dir> <output_dir> [analysis_level] [options]
```

### Example

```bash
python HBCD_ECG_pipeline_v6.py /path/to/bids /path/to/output participant_level \
  --participant-label 1234 \
  --session-label V01 \
  --tasks RS \
  --acq ecg
```

## Arguments

### Required positional arguments

- bids_dir: path to the BIDS dataset directory
- output_dir: path where results will be written
- analysis_level: optional positional; default is `participant_level` (only `participant_level` is supported)

### Optional filters

- `--participant-label`: one or more participant labels (without the `sub-` prefix)
- `--session-label`: one or more session labels (without the `ses-` prefix)

### Pipeline-specific options

- `--tasks`: one or more task names to process; default is `RS`
- `--acq`: acquisition label to match; default is `ecg`
- `--fallback-ecg-channel`: fallback ECG channel name if `channels.tsv` does not explicitly label ECG/EKG; default is `E128` (this option is present but suppressed from the help text)
- `--quality-threshold`: ECG quality cutoff for high-quality segment detection; default is `0.95`
- `--qc-window-sec`: seconds before/after event markers for QC plots; default is `10.0`

Notes:
- The script currently accepts only `participant_level` analysis (the positional `analysis_level` exists for BIDS-App compatibility).
- Task labels are normalized to uppercase by the script (e.g., `RS`).
- Participant and session label arguments must not contain path separators.

## Behavior notes

- The script currently implements participant-level analysis.
- group_level is accepted for BIDS-App compatibility but does not perform group-level processing.
- The pipeline uses task markers from events.tsv to define the analysis window, typically trimming from bas+ to TRSP.
- If the ECG channel is not explicitly labeled as ECG/EKG in channels.tsv, the script will use the fallback channel name.

## Output structure

For each processed file, results are written under:

```text
<output_dir>/<subject>/<session>/ecg/<task>-task/
  png/
  csv/
  data/
```

Typical output files include:

- PNG plots written to the `png` subfolder (examples saved by the pipeline):
  - `<title_base>_raw_ecg.png`
  - `<title_base>_filtered_ecg.png`
  - `<title_base>_delineation.png`
  - `<title_base>_heart_rate.png`
  - `<title_base>_rr_vs_hr.png`
  - `<title_base>_segmented_heartbeat.png`
  - `<title_base>_HRV_TimeDomain.png`
  - `<title_base>_HRV_new_metrics.png`
  - `<title_base>_filtered_task_window_isolated.png`
  - `<title_base>_filtered_task_window_isolated_zoom.png`

- CSV files written to the `csv` subfolder, including summary CSVs and RR-filter details. The script writes a participant/task summary as:
  - `<title_base>_summary.csv`

- Raw/derived `.set` and other data artifacts are written to the `data` subfolder, for example:
  - `<title_base>_raw_bas_trsp.set`
  - `<title_base>_filtered_ecg.set`

- Participant/session-level logs (written alongside the participant folder):
  - `skipped_no_ecg_channel_log.txt` — created when no ECG channel is selected for a participant/session
  - `failed_files_log.txt` — written under `<output_dir>/<subject>/<session>/failed_files_log.txt` when a file fails processing

Notes:
- `title_base` is typically the filename base such as `sub-1234_ses-V01_task-RS` and the pipeline constructs file names using this prefix.
- The pipeline creates `png`, `csv`, and `data` subfolders only after an ECG channel is successfully selected; skipped participants will contain the skipped log but no `png/csv/data` subfolders.

## Troubleshooting

If processing fails for a file:

- inspect the console output for the specific error
- review the failed_files_log.txt file in the base output directory
- verify that the ECG channel exists in the .set file and that the sidecar files are present and correctly formatted

## License

This project is licensed under the MIT License. See the LICENSE file for details.
