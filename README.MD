# HBCD ECG Pipeline

A BIDS-style ECG quality-control and analysis workflow for task-based EEG/ECG recordings.

This repository contains a processing script, HBCD_ECG_pipeline_v4.py, that reads EEGLab .set files, extracts an ECG channel, crops the signal to the task window, performs signal cleaning and R-peak analysis, and generates plots and CSV summaries for QC review.

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

Install them in your Python environment before running the script.

## Quick start

Run the pipeline from the repository directory:

```bash
python HBCD_ECG_pipeline_v4.py <bids_dir> <output_dir> <analysis_level> [options]
```

### Example

```bash
python HBCD_ECG_pipeline_v4.py /path/to/bids /path/to/output participant_level \
  --participant_label 240961 \
  --session_label V04 \
  --tasks RS \
  --acq ecg
```

## Arguments

### Required positional arguments

- bids_dir: path to the BIDS dataset directory
- output_dir: path where results will be written
- analysis_level: must be participant_level or group_level

### Optional filters

- --participant_label: one or more participant labels without the sub- prefix
- --session_label: one or more session labels without the ses- prefix

### Pipeline-specific options

- --tasks: one or more task names to process; default is RS
- --acq: acquisition label to match; default is ecg
- --fallback-ecg-channel: fallback ECG channel name if channels.tsv does not explicitly label ECG/EKG; default is E128
- --quality-threshold: threshold used for high-quality segment detection; default is 0.95
- --qc-window-sec: seconds before/after event markers shown in QC plots; default is 10

## Behavior notes

- The script currently implements participant-level analysis.
- group_level is accepted for BIDS-App compatibility but does not perform group-level processing.
- The pipeline uses task markers from events.tsv to define the analysis window, typically trimming from bas+ to TRSP.
- If the ECG channel is not explicitly labeled as ECG/EKG in channels.tsv, the script will use the fallback channel name.

## Output structure

For each processed file, results are written under:

```text
<output_dir>/<subject>/<session>/ecg/<task>-task/
```

Typical output files include:

- PNG plots for raw ECG, filtered ECG, quality, R-peaks, RR intervals, heart rate, HRV, segmentation, and task isolation
- CSV files containing RR interval filtering details and bad-segment information
- a summary CSV named like <subject>_<session>_task-<task>_summary.csv
- a failed_files_log.txt in the base output directory if any file fails during processing

## Example outputs

Examples of generated files include:

- <subject>_<session>_task-RS_raw_ecg.png
- <subject>_<session>_task-RS_filtered_ecg.png
- <subject>_<session>_task-RS_quality.png
- <subject>_<session>_task-RS_rpeak_validation.png
- <subject>_<session>_task-RS_heart_rate.png
- <subject>_<session>_task-RS_HRV_TimeDomain.png
- <subject>_<session>_task-RS_summary.csv

## Troubleshooting

If processing fails for a file:

- inspect the console output for the specific error
- review the failed_files_log.txt file in the base output directory
- verify that the ECG channel exists in the .set file and that the sidecar files are present and correctly formatted

## License

This project is provided as-is for internal HBCD ECG QC processing workflows.
