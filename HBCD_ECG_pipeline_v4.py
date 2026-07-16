### ECG Pipeline - HBCD ###

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # no display inside a container
import neurokit2 as nk
import mne
from scipy.signal import welch
from pathlib import Path
import traceback
import datetime
import argparse
import re
import warnings
warnings.filterwarnings(
    "ignore",
    message="The figure layout has changed to tight"
)

def parse_command_line_args():

    parser = argparse.ArgumentParser(
        description="Run the HBCD ECG task-based QC pipeline (BIDS-App)."
    )

    # ------------------------------------------------------------
    # BIDS-App required positional arguments
    # ------------------------------------------------------------
    parser.add_argument(
        'bids_dir',
        help='Path to the BIDS dataset directory'
    )

    parser.add_argument(
        'output_dir',
        help='Path to the output directory (will be created if it does not exist)'
    )

    parser.add_argument(
        'analysis_level',
        choices=['participant_level', 'group_level'],
        help='Level of analysis to perform'
    )

    # ------------------------------------------------------------
    # BIDS-App standard optional filters
    # ------------------------------------------------------------
    parser.add_argument(
        '--participant_label',
        nargs='+',
        dest='participant_labels',
        help='Space-separated list of participant labels to analyze (without "sub-" prefix). '
             'If not provided, all participants will be analyzed.'
    )

    parser.add_argument(
        '--session_label',
        nargs='+',
        dest='session_labels',
        help='Space-separated list of session labels to analyze (without "ses-" prefix). '
             'If not provided, all sessions for each participant will be analyzed.'
    )

    # ------------------------------------------------------------
    # Pipeline-specific optional arguments
    # ------------------------------------------------------------
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["RS"],
        help="Task(s) to process. Example: --tasks RS or --tasks RS MMN FACE VEP"
    )

    parser.add_argument(
        "--acq",
        default="ecg",
        help="Acquisition label to process. Usually use ecg to match acq-ecg files."
    )

    parser.add_argument(
        "--fallback-ecg-channel",
        default="E128",
        help="Backup ECG channel if channels.tsv does not explicitly label ECG/EKG."
    )

    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.95,
        help="ECG quality cutoff used for high-quality segment detection. Default is 0.95."
    )

    parser.add_argument(
        "--qc-window-sec",
        type=float,
        default=10,
        help="Seconds before/after event markers for QC plots. Default is 10."
    )

    args, unknown = parser.parse_known_args()

    if unknown:
        print("Note: ignoring extra command-line arguments:", unknown)

    return args


ARGS = parse_command_line_args()

# Expected input structure:
#   bids_dir/sub-240961/ses-V04/eeg/sub-240961_ses-V04_task-RS_acq-ecg_run-01_eeg.set
INPUT_DIR = Path(ARGS.bids_dir).expanduser().resolve()


# Expected output structure:
#   output_dir/sub-240961/ses-V04/ecg/rs-task/
OUTPUT_DIR = Path(ARGS.output_dir).expanduser().resolve()

# BIDS-App analysis level ('participant_level' or 'group_level').
ANALYSIS_LEVEL = ARGS.analysis_level

# Optional participant/session filters (labels WITHOUT the "sub-"/"ses-" prefix).
# None means "process everything found".
PARTICIPANT_LABELS = ARGS.participant_labels
SESSION_LABELS = ARGS.session_labels

TASKS_TO_PROCESS = [task.upper() for task in ARGS.tasks]

ACQ_TO_PROCESS = ARGS.acq

# If channels.tsv does not correctly label an ECG channel, use this as a backup.
FALLBACK_ECG_CHANNEL = ARGS.fallback_ecg_channel


DEFAULT_SAMPLING_RATE = 1000

QUALITY_THRESHOLD = ARGS.quality_threshold

# ------------------------------------------------------------
# PEAK CORRECTION / BAD-SEGMENT SETTINGS
# ------------------------------------------------------------
#
# 1. Peak correction uses an expected RR interval range:
#       interval_min = 0.30 seconds
#       interval_max = 0.75 seconds
#
# 2. After peak correction, bad segments are identified using:
#       abs(RR[n] - RR[n-1]) / RR[n] >= 0.20
#
FIXPEAKS_INTERVAL_MIN = 0.30
FIXPEAKS_INTERVAL_MAX = 0.75
RR_PERCENT_CHANGE_THRESHOLD = 0.20

# Add a small amount of time around each RR interval flagged by the 20% rule.
BAD_SEGMENT_PADDING_SEC = 1.0 # removes exactly the flagged interval with no buffer on either side.


# Prevent HR/RR interpolation across gaps larger than the upper RR threshold.
MAX_RR_FOR_INTERPOLATION = FIXPEAKS_INTERVAL_MAX

QC_WINDOW_SEC = ARGS.qc_window_sec
LONG_QC_WINDOW_SEC = 35
CUSTOM_QC_WINDOWS_SEC = [(50, 70)]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nHBCD ECG pipeline settings")
print("--------------------------")
print("BIDS directory:", INPUT_DIR)
print("Output folder:", OUTPUT_DIR)
print("Analysis level:", ANALYSIS_LEVEL)
print("Participant labels:", PARTICIPANT_LABELS if PARTICIPANT_LABELS else "ALL")
print("Session labels:", SESSION_LABELS if SESSION_LABELS else "ALL")
print("Tasks:", TASKS_TO_PROCESS)
print("Acquisition:", ACQ_TO_PROCESS)
print("Fallback ECG channel:", FALLBACK_ECG_CHANNEL)
print("Quality threshold:", QUALITY_THRESHOLD)
print("Peak-correction interval_min:", FIXPEAKS_INTERVAL_MIN)
print("Peak-correction interval_max:", FIXPEAKS_INTERVAL_MAX)
print("RR percent-change threshold:", RR_PERCENT_CHANGE_THRESHOLD)
print("QC marker window seconds:", QC_WINDOW_SEC)
print("--------------------------\n")

# ------------------------------------------------------------
# This pipeline only implements participant-level analysis.
# group_level is accepted (for BIDS-App compliance) but not implemented.
# ------------------------------------------------------------
if ANALYSIS_LEVEL == "group_level":
    print("Status: 'group_level' was requested, but this pipeline only implements 'participant_level' analysis.")
    print("Nothing to do. Exiting.")
    raise SystemExit(0)


# ------------------------------------------------------------
# FIGURE SETTINGS
# ------------------------------------------------------------
WIDE_FIG = (14, 5)
WIDE_TALL_FIG = (14, 6)
DOUBLE_FIG = (14, 9)
HRV_FIG = (16, 11)
TABLE_WIDTH = 16

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def simple_title(subject, session, task, suffix):
    """Keep titles simple, like sub-240961_ses-V04_task-RS: ..."""
    return f"Task-{task}: {suffix}"



def find_task_set_files(input_dir, tasks_to_process, acq_to_process="ecg", participant_labels=None, session_labels=None):
    """
    Find task .set files under input_dir, optionally restricted to specific
    participant labels (without "sub-") and/or session labels (without "ses-").
    """

    if participant_labels:
        subject_globs = [f"sub-{label}" for label in participant_labels]
    else:
        subject_globs = ["sub-*"]

    if session_labels:
        session_globs = [f"ses-{label}" for label in session_labels]
    else:
        session_globs = ["ses-*"]

    all_set_files = []
    for task in tasks_to_process:
        task_files = []
        for subject_glob in subject_globs:
            for session_glob in session_globs:
                pattern = f"{subject_glob}/{session_glob}/eeg/*task-{task}_acq-{acq_to_process}*eeg.set"
                task_files.extend(input_dir.glob(pattern))

        print(f"Search pattern(s) for task-{task}: participants={subject_globs}, sessions={session_globs}")
        print(f"Found {len(task_files)} task-{task} acq-{acq_to_process} .set files.\n")
        all_set_files.extend(task_files)

    # De-duplicate (e.g. if participant/session globs overlap) while keeping stable order.
    seen = set()
    unique_files = []
    for f in all_set_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return sorted(unique_files)



def get_matching_sidecars(set_file):
    set_file = Path(set_file)
    channels_tsv = Path(str(set_file).replace("_eeg.set", "_channels.tsv"))
    events_tsv = Path(str(set_file).replace("_eeg.set", "_events.tsv"))
    if not channels_tsv.exists():
        raise FileNotFoundError(f"Missing channels.tsv: {channels_tsv}")
    return channels_tsv, events_tsv



def get_ecg_channel_from_channels_tsv(channels_tsv, fallback_channel="E128"):
    channels = pd.read_csv(channels_tsv, sep="\t")
    channels.columns = channels.columns.str.strip()
    for col in channels.columns:
        if channels[col].dtype == "object":
            channels[col] = channels[col].astype(str).str.strip()

    if "name" not in channels.columns:
        raise ValueError(f"No 'name' column found in {channels_tsv}")

    if "type" in channels.columns:
        type_clean = channels["type"].astype(str).str.upper().str.strip()
        ecg_rows = channels[type_clean.isin(["ECG", "EKG"])]
        if not ecg_rows.empty:
            return ecg_rows["name"].iloc[0], "channels.tsv type column"

    if fallback_channel in channels["name"].values:
        return fallback_channel, f"fallback_{fallback_channel}_no_ECG_type_in_channels_tsv"

    raise ValueError(
        f"No ECG/EKG channel found and fallback channel {fallback_channel} not available in {channels_tsv}. "
        f"Available channels: {channels['name'].tolist()}"
    )



def make_output_folder(output_dir, set_file, task):
    """Save under ecg folder, not eeg."""
    set_file = Path(set_file)
    parts = set_file.parts
    subject = next(part for part in parts if part.startswith("sub-"))
    session = next(part for part in parts if part.startswith("ses-"))
    task_folder = f"{task.lower()}-task"
    output_folder = output_dir / subject / session / "ecg" / task_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder, subject, session


def save_placeholder_plot(output_path, title, message):
    fig, ax = plt.subplots(figsize=WIDE_FIG, constrained_layout=True)
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.03, 0.5, message, fontsize=12, va="center", wrap=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def finalize_and_save(fig, output_path):
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def figure_has_content(fig):
    """Return True if a Matplotlib figure has axes or visible content."""
    return fig is not None and len(fig.get_axes()) > 0



def choose_plot_window(signal, start=4000, end=6000, fallback=2000):
    if len(signal) <= start:
        return 0, min(len(signal), fallback)
    return start, min(len(signal), end)


def safe_minmax(arr):
    arr = np.asarray(arr)
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def read_events_file(events_tsv):
    if events_tsv is None or not Path(events_tsv).exists():
        return None
    try:
        events_df = pd.read_csv(events_tsv, sep='	')
        events_df.columns = events_df.columns.str.strip()
        if 'trial_type' in events_df.columns:
            events_df['trial_type'] = events_df['trial_type'].astype(str).str.strip()
        return events_df
    except Exception:
        return None


def get_task_start_end_markers(events_df):
    if events_df is None or len(events_df) == 0 or 'onset' not in events_df.columns:
        return None, None, None, None

    df = events_df.copy()
    if 'duration' not in df.columns:
        df['duration'] = 0
    if 'trial_type' not in df.columns:
        df['trial_type'] = ''

    # START priority: bas+ > DIN3 > last bgin > first non-boundary event
    start_sec = None
    start_label = None
    for marker in ['bas+', 'DIN3']:
        hits = df[df['trial_type'].str.lower() == marker.lower()]
        if len(hits) > 0:
            start_sec = float(hits.iloc[0]['onset'])
            start_label = marker
            break
    if start_sec is None:
        hits = df[df['trial_type'].str.lower() == 'bgin']
        if len(hits) > 0:
            start_sec = float(hits.iloc[-1]['onset'])
            start_label = 'bgin'
    if start_sec is None:
        hits = df[df['trial_type'].str.lower() != 'boundary']
        if len(hits) > 0:
            start_sec = float(hits.iloc[0]['onset'])
            start_label = str(hits.iloc[0]['trial_type'])

    # END priority: last TRSP > last event onset
    end_sec = None
    end_label = None
    hits = df[df['trial_type'].str.lower() == 'trsp']
    if len(hits) > 0:
        end_sec = float(hits.iloc[-1]['onset'])
        end_label = 'TRSP'
    else:
        end_sec = float(df.iloc[-1]['onset'])
        end_label = str(df.iloc[-1]['trial_type'])

    return start_sec, end_sec, start_label, end_label


def make_task_relative_events(events_df, task_start_sec, task_end_sec):
    """
    Keep only events inside the task window and reset onset so task_start_sec becomes 0.

    Example:
    - Full recording bas+ onset = 8.923 seconds
    - After trimming, bas+ onset becomes 0 seconds
    - TRSP becomes task_duration seconds

    This keeps plots from showing anything before bas+.
    """
    if events_df is None or len(events_df) == 0 or 'onset' not in events_df.columns:
        return None

    df = events_df.copy()
    df = df[
        (df['onset'] >= task_start_sec) &
        (df['onset'] <= task_end_sec)
    ].copy()

    df['onset'] = df['onset'] - task_start_sec

    return df.reset_index(drop=True)


def get_qc_spike_events(events_df):
    if events_df is None or len(events_df) == 0:
        return []
    df = events_df.copy()
    df.columns = df.columns.str.strip()
    if 'trial_type' not in df.columns or 'onset' not in df.columns:
        return []
    keep = ['boundary', 'VBeg', 'bgin', 'bas+', 'TRSP']
    out = df[df['trial_type'].astype(str).isin(keep)].copy()
    out = out.sort_values('onset').reset_index(drop=True)
    return out.to_dict('records')


def plot_first_n_seconds_after_task_start(signal, sampling_rate, n_seconds, subject, session, task, output_path, signal_name='Raw ECG'):
    """
    Plot the first n_seconds of the cropped task signal.

    Because the ECG was already cropped to bas+ -> TRSP, time 0 means bas+.
    So this plot shows the first n_seconds after bas+.
    """
    n = len(signal)
    end_sec = min(float(n_seconds), n / sampling_rate)
    end_idx = int(end_sec * sampling_rate)

    fig, ax = plt.subplots(figsize=WIDE_TALL_FIG, constrained_layout=True)
    ax.plot(np.arange(end_idx) / sampling_rate, signal[:end_idx], color='blue', lw=1.0)
    ax.set_title(simple_title(subject, session, task, f"{signal_name} First {int(n_seconds)} Seconds After bas+"))
    ax.set_xlabel("Task Time After bas+ (seconds)")
    ax.set_ylabel("Amplitude [mV]")
    ax.grid(True, alpha=0.3)
    finalize_and_save(fig, output_path)



def plot_custom_time_window(signal, sampling_rate, start_sec, end_sec, subject, session, task, output_path, signal_name='Filtered ECG'):
    n = len(signal)
    total_sec = n / sampling_rate
    start_sec = max(0, float(start_sec))
    end_sec = min(total_sec, float(end_sec))
    if end_sec <= start_sec:
        save_placeholder_plot(
            output_path,
            simple_title(subject, session, task, f"{signal_name} Custom Window"),
            f"Invalid custom window: start={start_sec}, end={end_sec}"
        )
        return

    start_idx = int(start_sec * sampling_rate)
    end_idx = int(end_sec * sampling_rate)
    times = np.arange(start_idx, end_idx) / sampling_rate

    fig, ax = plt.subplots(figsize=WIDE_TALL_FIG, constrained_layout=True)
    ax.plot(times, signal[start_idx:end_idx], color='blue', lw=1.0)
    ax.set_title(simple_title(subject, session, task, f"{signal_name} Custom Window {start_sec:.0f}-{end_sec:.0f} sec"))
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude [mV]')
    ax.grid(True, alpha=0.3)
    finalize_and_save(fig, output_path)


def plot_marker_qc(
    signal,
    sampling_rate,
    marker_sec,
    marker_label,
    subject,
    session,
    task,
    output_path,
    signal_name='Raw ECG',
    window_sec=10,
    events_df=None,
    event_labels_to_show=None
):

    if marker_sec is None:
        save_placeholder_plot(
            output_path,
            simple_title(subject, session, task, f"{signal_name} Marker QC"),
            "No marker found in events.tsv for this file."
        )
        return

    if event_labels_to_show is None:

        event_labels_to_show = ['boundary', 'VBeg', 'bgin', 'TRSP', 'bas+', 'DIN3']

    n = len(signal)
    start_sec = max(0, marker_sec - window_sec)
    end_sec = min(n / sampling_rate, marker_sec + window_sec)
    start_idx = int(start_sec * sampling_rate)
    end_idx = int(end_sec * sampling_rate)
    times = np.arange(start_idx, end_idx) / sampling_rate
    signal_window = signal[start_idx:end_idx]

    fig, ax = plt.subplots(figsize=WIDE_TALL_FIG, constrained_layout=True)
    ax.plot(times, signal_window, color='blue', lw=1.0)

    if len(signal_window) > 0:
        y_top = np.nanmax(signal_window)
        y_bottom = np.nanmin(signal_window)
        y_range = y_top - y_bottom if y_top != y_bottom else 1
    else:
        y_top, y_bottom, y_range = 1, 0, 1

    if events_df is not None and 'onset' in events_df.columns:
        nearby_events = events_df[
            (events_df['onset'] >= start_sec) &
            (events_df['onset'] <= end_sec)
        ].copy()

        if 'trial_type' in nearby_events.columns:
            nearby_events = nearby_events[
                nearby_events['trial_type'].astype(str).isin(event_labels_to_show)
            ]

        nearby_events = nearby_events.sort_values('onset').reset_index(drop=True)

        close_event_window_sec = 0.45
        event_groups = []

        for _, row in nearby_events.iterrows():
            event_sec = float(row['onset'])
            event_label = str(row['trial_type']) if 'trial_type' in row else 'event'

            ax.axvline(
                event_sec,
                color='gray',
                linestyle=':',
                linewidth=1.0,
                alpha=0.75
            )

            if len(event_groups) == 0 or event_sec - event_groups[-1]['last_sec'] > close_event_window_sec:
                event_groups.append({
                    'secs': [event_sec],
                    'labels': [event_label],
                    'last_sec': event_sec
                })
            else:
                event_groups[-1]['secs'].append(event_sec)
                event_groups[-1]['labels'].append(event_label)
                event_groups[-1]['last_sec'] = event_sec

        label_levels = [0.92, 0.74, 0.56, 0.38]
        for group_i, group in enumerate(event_groups):
            group_sec = float(np.mean(group['secs']))
            y_text = y_bottom + label_levels[group_i % len(label_levels)] * y_range


            group_text = "\n".join([
                f"{lab} {sec:.3f}s"
                for lab, sec in zip(group['labels'], group['secs'])
            ])

            ax.annotate(
                group_text,
                xy=(group_sec, y_text),
                xytext=(10, 0),
                textcoords='offset points',
                ha='left',
                va='center',
                fontsize=8,
                color='dimgray',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.75)
            )

   
    ax.axvline(
        marker_sec,
        color='red',
        linestyle='--',
        linewidth=1.8,
        label=f'Selected: {marker_label} @ {marker_sec:.3f}s'
    )

    ax.set_title(
        simple_title(
            subject,
            session,
            task,
            f"{signal_name} QC Around {marker_label} (±{window_sec} sec)"
        )
    )
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude [mV]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    finalize_and_save(fig, output_path)


def save_rs_summary_csv(output_folder, title_base, summary_rows):
    summary_df = pd.DataFrame(summary_rows, columns=['metric', 'value'])
    summary_df.to_csv(Path(output_folder) / f"{title_base}_summary.csv", index=False)


# Compare R-peaks before and after signal_fixpeaks correction.

def plot_peak_correction_comparison(ecg_filtered, original_peaks, cleaned_peaks, sampling_rate, title_base, output_folder, task_label):
    if len(original_peaks) < 2 or len(cleaned_peaks) < 2:
        save_placeholder_plot(
            os.path.join(output_folder, f"{title_base}_peak_correction_comparison.png"),
            simple_title(*task_label, "Peak Correction Comparison (Original vs Cleaned)"),
            "Not enough peaks to create original vs cleaned comparison plot."
        )
        return

   
    all_ref = cleaned_peaks if len(cleaned_peaks) > 0 else original_peaks
    first_idx = max(0, all_ref[0] - 200)
    ref_end_peak = all_ref[min(len(all_ref) - 1, 4)]
    last_idx = min(len(ecg_filtered), ref_end_peak + 300)
    x = np.arange(first_idx, last_idx)

    fig, axes = plt.subplots(2, 1, figsize=DOUBLE_FIG, sharex=True, constrained_layout=True)
    axes[0].plot(x, ecg_filtered[first_idx:last_idx], color='blue', lw=1.0, label='Filtered ECG')
    use_orig = original_peaks[(original_peaks >= first_idx) & (original_peaks <= last_idx)]
    if len(use_orig) > 0:
        axes[0].scatter(use_orig, ecg_filtered[use_orig], color='red', s=32, label='Original R-peaks', zorder=3)
    axes[0].set_title(simple_title(*task_label, "Original R-Peaks (Before Correction)"), fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Voltage [mV]")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, ecg_filtered[first_idx:last_idx], color='blue', lw=1.0, label='Filtered ECG')
    use_clean = cleaned_peaks[(cleaned_peaks >= first_idx) & (cleaned_peaks <= last_idx)]
    if len(use_clean) > 0:
        axes[1].scatter(use_clean, ecg_filtered[use_clean], color='green', s=32, label='Cleaned R-peaks', zorder=3)
    axes[1].set_title(simple_title(*task_label, "Cleaned R-Peaks (After Correction)"), fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Samples [N]")
    axes[1].set_ylabel("Voltage [mV]")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(simple_title(*task_label, "Peak Correction Comparison (Original vs Cleaned)"), fontsize=13, fontweight='bold')
    finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_peak_correction_comparison.png"))



def apply_20percent_rr_change_filter(peaks, sampling_rate, threshold=0.20):
    """
    Flag an RR interval when the change from the previous RR interval is at least 20%.

    Formula requested by the team:
        abs(RR[n] - RR[n-1]) / RR[n] >= threshold

    Notes:
    - RR[n] is the current RR interval.
    - RR[n-1] is the previous RR interval.
    - Absolute value flags both sudden increases and sudden decreases.
    - The first RR interval cannot be evaluated because it has no previous RR.
    """
    peaks = np.asarray(peaks, dtype=int)

    if len(peaks) < 2:
        rr_raw = np.array([])
        rr_cleaned = np.array([])
        valid_mask = np.array([], dtype=bool)
        invalid_indices = np.array([], dtype=int)
        percent_change = np.array([])
        metrics = {
            "total_raw_rr_intervals": 0,
            "rr_intervals_removed_20percent": 0,
            "rr_filter_yield_pct": 0.0,
            "mean_rri_20percent_cleaned": np.nan,
            "std_rri_20percent_cleaned": np.nan
        }
        return rr_raw, rr_cleaned, valid_mask, invalid_indices, percent_change, metrics

    rr_raw = np.diff(peaks) / sampling_rate
    valid_mask = np.ones(len(rr_raw), dtype=bool)

    if len(rr_raw) > 1:
        current_rr = rr_raw[1:]
        previous_rr = rr_raw[:-1]

        percent_change = np.full(len(current_rr), np.nan, dtype=float)
        safe = np.isfinite(current_rr) & np.isfinite(previous_rr) & (current_rr > 0)
        percent_change[safe] = (
            np.abs(current_rr[safe] - previous_rr[safe]) / current_rr[safe]
        )

        # +1 maps each comparison back to the CURRENT RR interval, RR[n].
        invalid_indices = np.where(percent_change >= threshold)[0] + 1
        valid_mask[invalid_indices] = False
    else:
        percent_change = np.array([])
        invalid_indices = np.array([], dtype=int)

    rr_cleaned = rr_raw[valid_mask]
    total_raw_rr = len(rr_raw)
    rr_removed = len(invalid_indices)
    filter_yield_pct = (
        ((total_raw_rr - rr_removed) / total_raw_rr) * 100
        if total_raw_rr > 0 else 0.0
    )

    metrics = {
        "total_raw_rr_intervals": int(total_raw_rr),
        "rr_intervals_removed_20percent": int(rr_removed),
        "rr_filter_yield_pct": float(filter_yield_pct),
        "mean_rri_20percent_cleaned": (
            float(np.mean(rr_cleaned)) if len(rr_cleaned) > 0 else np.nan
        ),
        "std_rri_20percent_cleaned": (
            float(np.std(rr_cleaned)) if len(rr_cleaned) > 0 else np.nan
        )
    }

    return rr_raw, rr_cleaned, valid_mask, invalid_indices, percent_change, metrics

def merge_bad_segments(segments):
    """
    Merge bad segments that overlap or touch each other.
    This avoids having many tiny overlapping red shaded regions.
    """

    if len(segments) == 0:
        return []

    segments = sorted(segments, key=lambda x: x["start_sec"])
    merged = [segments[0].copy()]

    for seg in segments[1:]:
        last = merged[-1]

        if seg["start_sec"] <= last["end_sec"]:
            last["end_sec"] = max(last["end_sec"], seg["end_sec"])
            last["end_sample"] = max(last["end_sample"], seg["end_sample"])
            last["duration_sec"] = last["end_sec"] - last["start_sec"]
            last["reason"] = last["reason"] + ";" + seg["reason"]
        else:
            merged.append(seg.copy())

    return merged


def make_bad_segments_from_20percent_rr(
    peaks,
    invalid_rr_indices,
    sampling_rate,
    signal_length,
    padding_sec=1.0
):
    """
    Convert RR intervals flagged by the 20% rule into ECG bad segments.

    RR interval i spans peaks[i] to peaks[i+1]. The corresponding ECG interval is
    marked as bad, with optional padding added before and after the interval.
    Overlapping bad segments are merged by merge_bad_segments().
    """
    peaks = np.asarray(peaks, dtype=int)
    bad_segments = []

    for rr_i in invalid_rr_indices:
        if rr_i < 0 or rr_i >= len(peaks) - 1:
            continue

        start_sample = int(max(0, peaks[rr_i] - padding_sec * sampling_rate))
        end_sample = int(min(
            signal_length,
            peaks[rr_i + 1] + padding_sec * sampling_rate
        ))

        if end_sample <= start_sample:
            continue

        bad_segments.append({
            "start_sample": start_sample,
            "end_sample": end_sample,
            "start_sec": start_sample / sampling_rate,
            "end_sec": end_sample / sampling_rate,
            "duration_sec": (end_sample - start_sample) / sampling_rate,
            "reason": "rr_change_at_least_20percent"
        })

    return merge_bad_segments(bad_segments)

def remove_peaks_in_bad_segments(peaks, bad_segments, sampling_rate):
    """
    Remove R-peaks that fall inside bad ECG segments.

    """

    peaks = np.asarray(peaks, dtype=int)

    if len(peaks) == 0 or len(bad_segments) == 0:
        return peaks

    keep = np.ones(len(peaks), dtype=bool)
    peak_times = peaks / sampling_rate

    for seg in bad_segments:
        keep = keep & ~((peak_times >= seg["start_sec"]) & (peak_times <= seg["end_sec"]))

    return peaks[keep]


def rr_intervals_excluding_bad_segments(peaks, bad_segments, sampling_rate, max_rr_sec=None):


    peaks = np.asarray(peaks, dtype=int)

    if len(peaks) < 2:
        return np.array([]), np.array([]), np.array([], dtype=bool), np.array([]), np.array([])

    rr_all = np.diff(peaks) / sampling_rate
    rr_times = peaks[1:] / sampling_rate
    valid_mask = np.ones(len(rr_all), dtype=bool)

    if max_rr_sec is not None:
        valid_mask = valid_mask & (rr_all <= max_rr_sec)

    for i in range(len(rr_all)):
        interval_start = peaks[i] / sampling_rate
        interval_end = peaks[i + 1] / sampling_rate

        for seg in bad_segments:
            overlaps_bad_segment = interval_start < float(seg["end_sec"]) and interval_end > float(seg["start_sec"])
            if overlaps_bad_segment:
                valid_mask[i] = False
                break

    rr_valid = rr_all[valid_mask]
    rr_times_valid = rr_times[valid_mask]

    return rr_all, rr_times, valid_mask, rr_valid, rr_times_valid


def build_pseudo_peaks_from_rr(rr_intervals_sec, sampling_rate):
    """
    - keeps only the final clean RR intervals.
    """

    rr_intervals_sec = np.asarray(rr_intervals_sec, dtype=float)
    rr_intervals_sec = rr_intervals_sec[np.isfinite(rr_intervals_sec)]

    if len(rr_intervals_sec) < 2:
        return np.array([], dtype=int)

    cumulative_sec = np.concatenate([[0], np.cumsum(rr_intervals_sec)])
    pseudo_peaks = np.round(cumulative_sec * sampling_rate).astype(int)

    return pseudo_peaks


def compute_safe_interpolated_hr(peaks, sampling_rate, signal_length, bad_segments=None, max_gap_sec=0.75):
    """
    Compute a heart-rate line from peaks, but leave gaps over bad segments.

    """

    peaks = np.asarray(peaks, dtype=int)
    ecg_rate = np.full(signal_length, np.nan)

    if len(peaks) < 2:
        return ecg_rate

    rr = np.diff(peaks) / sampling_rate
    hr_values = 60 / rr
    hr_times = peaks[1:] / sampling_rate

    for i in range(len(hr_values) - 1):
        start_time = hr_times[i]
        end_time = hr_times[i + 1]
        gap = end_time - start_time

        if max_gap_sec is not None and gap > max_gap_sec:
            continue

        start_sample = int(start_time * sampling_rate)
        end_sample = int(end_time * sampling_rate)

        if end_sample <= start_sample:
            continue

        ecg_rate[start_sample:end_sample] = np.linspace(hr_values[i], hr_values[i + 1], end_sample - start_sample)

    if bad_segments is not None:
        for seg in bad_segments:
            start_sample = int(max(0, seg["start_sample"]))
            end_sample = int(min(signal_length, seg["end_sample"]))
            ecg_rate[start_sample:end_sample] = np.nan

    return ecg_rate


def compute_safe_interpolated_hr_from_rr(
    rr_times,
    rr_intervals,
    sampling_rate,
    signal_length,
    bad_segments=None,
    max_gap_sec=0.75
):
    """
    Compute heart rate from FINAL cleaned RR intervals.

    """

    rr_times = np.asarray(rr_times, dtype=float)
    rr_intervals = np.asarray(rr_intervals, dtype=float)

    ecg_rate = np.full(signal_length, np.nan)

    keep = np.isfinite(rr_times) & np.isfinite(rr_intervals) & (rr_intervals > 0)
    rr_times = rr_times[keep]
    rr_intervals = rr_intervals[keep]

    if len(rr_intervals) == 0:
        return ecg_rate

    hr_values = 60 / rr_intervals

    # If there is only one valid RR interval, place one HR point but do not interpolate.
    if len(hr_values) == 1:
        sample = int(rr_times[0] * sampling_rate)
        if 0 <= sample < signal_length:
            ecg_rate[sample] = hr_values[0]
    else:
        for i in range(len(hr_values) - 1):
            start_time = rr_times[i]
            end_time = rr_times[i + 1]
            gap = end_time - start_time

            # This is the key safety step:
            # Do not draw a continuous HR line across a removed/bad segment.
            if max_gap_sec is not None and gap > max_gap_sec:
                continue

            start_sample = int(start_time * sampling_rate)
            end_sample = int(end_time * sampling_rate)

            start_sample = max(0, min(signal_length, start_sample))
            end_sample = max(0, min(signal_length, end_sample))

            if end_sample <= start_sample:
                continue

            ecg_rate[start_sample:end_sample] = np.linspace(
                hr_values[i],
                hr_values[i + 1],
                end_sample - start_sample
            )

    # Force removed ECG sections to stay missing.
    if bad_segments is not None:
        for seg in bad_segments:
            start_sample = int(max(0, seg["start_sample"]))
            end_sample = int(min(signal_length, seg["end_sample"]))
            ecg_rate[start_sample:end_sample] = np.nan

    return ecg_rate


def nan_summary(arr):
    """
    Mean/min/max/std while ignoring NaN gaps.
    """
    arr = np.asarray(arr, dtype=float)
    if np.all(np.isnan(arr)):
        return np.nan, np.nan, np.nan, np.nan
    return float(np.nanmean(arr)), float(np.nanmin(arr)), float(np.nanmax(arr)), float(np.nanstd(arr))



# ------------------------------------------------------------
# FIND FILES
# ------------------------------------------------------------
file_paths = find_task_set_files(
    INPUT_DIR,
    TASKS_TO_PROCESS,
    ACQ_TO_PROCESS,
    participant_labels=PARTICIPANT_LABELS,
    session_labels=SESSION_LABELS
)
if not file_paths:
    print(f"Status: No .set files found in {INPUT_DIR}")
    if PARTICIPANT_LABELS:
        print(f"  (filtered to participant labels: {PARTICIPANT_LABELS})")
    if SESSION_LABELS:
        print(f"  (filtered to session labels: {SESSION_LABELS})")
    raise SystemExit(1)

total_files = len(file_paths)

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------

for index, file_path in enumerate(file_paths, start=1):
    filename = os.path.basename(file_path)
    SUBJECT_ID_BASE = os.path.splitext(filename)[0].replace("_eeg", "")
    task = filename.split("_task-")[1].split("_")[0]

    print(f"\n{'='*70}")
    print(f"Processing File [{index}/{total_files}]: {filename}")
    print(f"Task: task-{task}")
    print(f"{'='*70}")

    try:
        output_folder, subject, session = make_output_folder(OUTPUT_DIR, file_path, task)
        print(f"Saving outputs to: {output_folder}")

        channels_tsv, events_tsv = get_matching_sidecars(file_path)
        ecg_channel, ecg_channel_source = get_ecg_channel_from_channels_tsv(channels_tsv, FALLBACK_ECG_CHANNEL)
        print(f"Using ECG channel: {ecg_channel}")
        print(f"ECG channel source: {ecg_channel_source}")

        events_df = read_events_file(events_tsv)
        marker_start_sec, marker_end_sec, marker_start_label, marker_end_label = get_task_start_end_markers(events_df)


        ########################## July 13, 2026 / Define bgin marker #############################
        bgin_events = events_df[events_df['trial_type'] == 'bgin'] if events_df is not None else []
        if len(bgin_events) >= 2:
            bgin_marker_sec = float(bgin_events['onset'].values[1]) # Grabs the second bgin variable
            print(f"Found bgin marker before bas+ at: {bgin_marker_sec:.3f}s")
        elif len(bgin_events) == 1:
            bgin_marker_sec = float(bgin_events['onset'].values[0]) # Grabs the first bgin variable
            print(f"Found only one bgin marker at: {bgin_marker_sec:.3f}s")
        else:
            bgin_marker_sec = None
            print("No bgin marker found in this events file.")

        # Identify DIN3 marker for the full-session isolation plots.
        din3_events = (
            events_df[events_df["trial_type"].astype(str).str.lower() == "din3"]
            if events_df is not None and "trial_type" in events_df.columns
            else []
        )
        if len(din3_events) > 0:
            din3_marker_sec = float(din3_events["onset"].values[0])
            print(f"Found DIN3 marker at: {din3_marker_sec:.3f}s")
        else:
            din3_marker_sec = None
            print("No DIN3 marker found in this events file.")


        ######################################################################


        if marker_start_sec is not None:
            print(f"Start marker: {marker_start_label} @ {marker_start_sec:.3f}s")
        if marker_end_sec is not None:
            print(f"End marker: {marker_end_label} @ {marker_end_sec:.3f}s")

        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        sampling_rate = int(raw.info["sfreq"]) if raw.info.get("sfreq") else DEFAULT_SAMPLING_RATE

        if ecg_channel not in raw.ch_names:
            raise ValueError(f"ECG channel {ecg_channel} not found in .set file. Available channels: {raw.ch_names}")

        # Pull the ECG channel from the .set file.
        ecg_raw_full = raw.get_data(picks=[ecg_channel])[0].flatten()
        full_duration_sec = len(ecg_raw_full) / sampling_rate


        # keep only the actual task period.
        #
        # For task-RS, the task period is:
        #   start = bas+
        #   end   = TRSP
        #
        # After this point:
        #   ecg_raw = cropped task-only signal
        #   time 0  = bas+
        #   the final sample is around TRSP
        #
        task_start_sec = marker_start_sec if marker_start_sec is not None else 0
        task_end_sec = marker_end_sec if marker_end_sec is not None else full_duration_sec

        task_start_sec = max(0, float(task_start_sec))
        task_end_sec = min(full_duration_sec, float(task_end_sec))

        if task_end_sec <= task_start_sec:
            raise ValueError(
                f"Invalid task window: start={task_start_sec}, end={task_end_sec}. "
                "Check bas+ and TRSP markers in events.tsv."
            )

        task_start_sample = int(task_start_sec * sampling_rate)
        task_end_sample = int(task_end_sec * sampling_rate)

        # This is now the ONLY ECG signal used for the rest of the pipeline.
        ecg_raw = ecg_raw_full[task_start_sample:task_end_sample]

        # Keep only task-window events and reset their onsets so bas+ becomes 0 seconds.
        events_df_task = make_task_relative_events(events_df, task_start_sec, task_end_sec)

        # After cropping, the selected start/end markers are relative to the task signal.
        marker_start_sec_task = 0
        marker_start_label_task = marker_start_label if marker_start_label is not None else "task_start"

        marker_end_sec_task = len(ecg_raw) / sampling_rate
        marker_end_label_task = marker_end_label if marker_end_label is not None else "task_end"

        title_base = f"{subject}_{session}_task-{task}"
        task_label = (subject, session, task)

        print(f"Full recording samples: {len(ecg_raw_full)}")
        print(f"Full recording duration: {full_duration_sec:.1f} seconds")
        print(f"Analysis window in original recording: {task_start_sec:.3f}s to {task_end_sec:.3f}s")
        print(f"After cropping, task time starts at 0.000s and ends at {marker_end_sec_task:.3f}s")
        print(f"Task samples after cropping: {len(ecg_raw)}")
        print(f"Task duration after cropping: {len(ecg_raw)/sampling_rate:.1f} seconds")


        # This plot shows ALL raw ECG data after cropping to bas+ -> TRSP.
        raw_times = np.arange(len(ecg_raw)) / sampling_rate
        fig, ax = plt.subplots(figsize=WIDE_FIG, constrained_layout=True)
        ax.plot(raw_times, ecg_raw, color='blue', lw=0.8)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_title(simple_title(subject, session, task, "Unfiltered Raw ECG (Full Task Window: bas+ to TRSP)"))
        ax.set_xlabel("Task Time After bas+ (seconds)")
        ax.set_ylabel("Amplitude [mV]")
        ax.grid(True, alpha=0.3)
        finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_raw_ecg.png"))

        # 1a. Raw ECG - first 25 seconds after bas+
        plot_first_n_seconds_after_task_start(
            ecg_raw,
            sampling_rate,
            25,
            subject,
            session,
            task,
            os.path.join(output_folder, f"{title_base}_raw_ecg_first25sec_after_bas.png"),
            signal_name="Raw ECG"
        )

        # 2. Raw PSD
        try:
            nk.signal_psd(ecg_raw, method="welch", min_frequency=1, show=True)
            fig = plt.gcf()
            fig.set_size_inches(*WIDE_TALL_FIG)
            ax = fig.axes[0]
            ax.set_title(simple_title(subject, session, task, "Power Spectral Density - Unfiltered ECG"))
            ax.set_xlim(0, 100)
            ax.axvline(60, color='red', linestyle='--', label='60 Hz Powerline')
            ax.legend(loc='upper right')
            finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_raw_psd.png"))
        except Exception as e:
            save_placeholder_plot(
                os.path.join(output_folder, f"{title_base}_raw_psd.png"),
                simple_title(subject, session, task, "Power Spectral Density - Unfiltered ECG"),
                f"Could not compute raw PSD: {e}"
            )

        # 3. Filter
        # Filtering removes slow drift, high-frequency noise, and 60 Hz electrical noise.
        # This is already bas+ to TRSP only.
        ecg_filtered = nk.signal_filter(
            ecg_raw,
            sampling_rate=sampling_rate,
            lowcut=1,
            highcut=40,
            method='butterworth',
            order=2,
            powerline=60
        )

        # 4. Filtered ECG
        fig, ax = plt.subplots(figsize=WIDE_FIG, constrained_layout=True)
        start, end = choose_plot_window(ecg_filtered)
        ax.plot(np.arange(start, end), ecg_filtered[start:end], color='red', lw=0.9,
                label='Filtered (60 Hz Notch + 1-40 Hz Bandpass)')
        ax.set_title(simple_title(subject, session, task, "Filtered ECG"))
        ax.set_xlabel("Samples [N]")
        ax.set_ylabel("Voltage [mV]")
        ax.legend(loc='upper right')
        finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_filtered_ecg.png"))



        # 5. Quality plot with threshold line
        # Higher values mean the signal looks more like clean ECG.
        quality = np.asarray(nk.ecg_quality(ecg_filtered, sampling_rate=sampling_rate, method="templatematch")).flatten()
        fig, ax = plt.subplots(figsize=WIDE_FIG, constrained_layout=True)
        if len(quality) > 1:
            ax.plot(quality, lw=1.0)
        else:
            ax.scatter([0], quality, color='red')
        ax.axhline(QUALITY_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
                   label=f'Threshold = {QUALITY_THRESHOLD}')
        ax.set_title(simple_title(subject, session, task, "ECG Quality (Template Match)"))
        ax.set_xlabel("Samples [N]")
        ax.set_ylabel("Correlation Coefficient [r]")
        qmin, qmax = safe_minmax(quality)
        ax.set_ylim(qmin - 0.05 * (1.0 - qmin), 1.0)
        ax.legend(loc='lower right')
        finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_quality.png"))

        # Quality summary
        min_val = np.min(quality)
        n_worst = min(5, len(quality))
        worst_indices = np.argpartition(quality, n_worst - 1)[:n_worst] if n_worst > 0 else []
        worst_values = quality[worst_indices] if n_worst > 0 else []
        high_quality_pct = (np.sum(quality >= QUALITY_THRESHOLD) / len(quality)) * 100

        print("--- Quality Report: ---")
        print(f"Absolute Minimum Quality: {min_val:.4f}")
        print(f"Absolute Maximum Quality: {np.max(quality):.4f}")
        print(f"Mean Quality: {np.mean(quality):.4f}")
        print(f"Standard Deviation: {np.std(quality):.4f}")
        print(f"Percentage of file > {QUALITY_THRESHOLD} quality: {high_quality_pct:.2f}%")

        # 6. High-quality segments plot
        try:
            events = nk.events_find(quality, threshold=QUALITY_THRESHOLD, threshold_keep='above', duration_min=sampling_rate * 30)
            onsets = np.asarray(events['onset'])
            offsets = np.asarray(events['onset']) + np.asarray(events['duration'])
            nk.events_plot([onsets, offsets], quality)
            fig = plt.gcf()
            fig.set_size_inches(*WIDE_FIG)
            ax = fig.axes[0]

            handles, labels = ax.get_legend_handles_labels()
            new_labels = []
            for lab in labels:
                if lab == "0":
                    new_labels.append("Onset")
                elif lab == "1":
                    new_labels.append("Offset")
                else:
                    new_labels.append(lab)
            if handles:
                ax.legend(handles, new_labels, loc="lower right")

            ax.set_title(simple_title(subject, session, task, f"Detected High-Quality Segments (r > {QUALITY_THRESHOLD})"))
            ax.set_xlabel("Samples [N]")
            ax.set_ylabel("Correlation Coefficient [r]")
            finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_high_quality_segments.png"))
        except Exception as e:
            events = {'onset': np.array([]), 'duration': np.array([])}
            onsets = np.array([])
            offsets = np.array([])
            save_placeholder_plot(
                os.path.join(output_folder, f"{title_base}_high_quality_segments.png"),
                simple_title(subject, session, task, f"Detected High-Quality Segments (r > {QUALITY_THRESHOLD})"),
                f"Could not identify/plot high-quality segments: {e}"
            )

        # Segment report
        total_usable_seconds = sum(events['duration']) / sampling_rate if len(events['duration']) > 0 else 0

        # 7. Peaks
        rpeaks, info = nk.ecg_peaks(ecg_filtered, sampling_rate=sampling_rate)
        original_peaks = np.asarray(info['ECG_R_Peaks'], dtype=int)
        rr_intervals_sec = np.diff(original_peaks) / sampling_rate if len(original_peaks) > 1 else np.array([])

        # R-peak validation plot
        try:
            if len(original_peaks) > 1:
                plot_start, plot_end = choose_plot_window(ecg_filtered, start=4000, end=5000, fallback=1000)
                x = np.arange(plot_start, plot_end)

                peaks_in_window = original_peaks[
                    (original_peaks >= plot_start) & (original_peaks < plot_end)
                ]

                fig, ax = plt.subplots(figsize=WIDE_FIG, constrained_layout=True)
                ax.plot(
                    x - plot_start,
                    ecg_filtered[plot_start:plot_end],
                    color='blue',
                    lw=1.0,
                    label='Filtered ECG'
                )

                if len(peaks_in_window) > 0:
                    ax.scatter(
                        peaks_in_window - plot_start,
                        ecg_filtered[peaks_in_window],
                        color='red',
                        s=32,
                        label='R-peaks',
                        zorder=3
                    )

                ax.set_title(simple_title(subject, session, task, "R-Peak Detection Validation"))
                ax.set_xlabel("Samples [N]")
                ax.set_ylabel("Voltage [mV]")
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_rpeak_validation.png"))
            else:
                save_placeholder_plot(
                    os.path.join(output_folder, f"{title_base}_rpeak_validation.png"),
                    simple_title(subject, session, task, "R-Peak Detection Validation"),
                    "Not enough R-peaks detected to plot validation."
                )
        except Exception as e:
            save_placeholder_plot(
                os.path.join(output_folder, f"{title_base}_rpeak_validation.png"),
                simple_title(subject, session, task, "R-Peak Detection Validation"),
                f"Could not plot R-peaks: {e}"
            )

        # 8. RR intervals from original peaks
        fig, ax = plt.subplots(figsize=WIDE_FIG, constrained_layout=True)
        if len(rr_intervals_sec) > 0:
            ax.plot(rr_intervals_sec, marker='o', linestyle='-', lw=1.0, markersize=4)
        ax.set_title(simple_title(subject, session, task, "RR Intervals from Original R-Peaks"))
        ax.set_xlabel("Beat Number")
        ax.set_ylabel("RR Interval (seconds)")
        ax.set_ylim(0, 2)
        finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_rr_intervals.png"))

        # 9. Peak correction with signal_fixpeaks
        # RR interval limits are applied during peak correction before 20% relative-change rule.
        try:
            info_fix, peaks_cleaned = nk.signal_fixpeaks(
                original_peaks,
                sampling_rate=sampling_rate,
                method="neurokit",
                iterative=True,
                interval_min=FIXPEAKS_INTERVAL_MIN,
                interval_max=FIXPEAKS_INTERVAL_MAX,
                show=False
            )
        except Exception as e:
            peaks_cleaned = original_peaks
            info_fix = {}
            print(f"WARNING: signal_fixpeaks failed; using original peaks. Error: {e}")

        peaks_cleaned = np.asarray(peaks_cleaned, dtype=int)

        # 9b. RR intervals from cleaned peaks + 20% relative-change rule
        (
            rr_raw,
            rr_cleaned_20pct,
            rr_valid_mask_20pct,
            invalid_indices,
            rr_percent_change,
            rr20_metrics
        ) = apply_20percent_rr_change_filter(
            peaks_cleaned,
            sampling_rate=sampling_rate,
            threshold=RR_PERCENT_CHANGE_THRESHOLD
        )

        total_raw_beats = rr20_metrics["total_raw_rr_intervals"]
        beats_removed = rr20_metrics["rr_intervals_removed_20percent"]
        filter_yield_pct = rr20_metrics["rr_filter_yield_pct"]
        mean_rri = rr20_metrics["mean_rri_20percent_cleaned"]
        std_rri = rr20_metrics["std_rri_20percent_cleaned"]

        print(
            f"Number of RR intervals after cleaning: {beats_removed} intervals "
            f"violated the {RR_PERCENT_CHANGE_THRESHOLD:.0%} change rule. "
            f"Remaining intervals: {len(rr_cleaned_20pct)}"
        )

        # Save one row per RR interval so the rule can be audited later.
        if len(rr_raw) > 0:
            percent_change_full = np.full(len(rr_raw), np.nan)
            if len(rr_percent_change) > 0:
                percent_change_full[1:] = rr_percent_change

            rr20_table = pd.DataFrame({
                "rr_index": np.arange(len(rr_raw)),
                "rr_interval_sec": rr_raw,
                "percent_change_from_previous_rr": percent_change_full,
                "kept_after_20percent_rule": rr_valid_mask_20pct
            })
            rr20_table.to_csv(
                os.path.join(
                    output_folder,
                    f"{title_base}_rr_20percent_change_filter.csv"
                ),
                index=False
            )

        # 9c. Convert RR intervals flagged by the 20% rule into bad ECG segments.
        bad_segments = make_bad_segments_from_20percent_rr(
            peaks=peaks_cleaned,
            invalid_rr_indices=invalid_indices,
            sampling_rate=sampling_rate,
            signal_length=len(ecg_filtered),
            padding_sec=BAD_SEGMENT_PADDING_SEC
        )

        bad_segments_df = pd.DataFrame(bad_segments)
        bad_segments_df.to_csv(
            os.path.join(
                output_folder,
                f"{title_base}_bad_segments_20percent.csv"
            ),
            index=False
        )

        peaks_final = remove_peaks_in_bad_segments(
            peaks_cleaned,
            bad_segments,
            sampling_rate
        )

        rr_all_after_gap_mask, rr_times_all_after_gap_mask, rr_gap_valid_mask, rr_final_cleaned, rr_final_times = rr_intervals_excluding_bad_segments(
            peaks=peaks_final,
            bad_segments=bad_segments,
            sampling_rate=sampling_rate,
            max_rr_sec=MAX_RR_FOR_INTERPOLATION
        )

        peaks_for_hrv = build_pseudo_peaks_from_rr(rr_final_cleaned, sampling_rate)

        if len(rr_all_after_gap_mask) > 0:
            rr_gap_table = pd.DataFrame({
                "rr_index": np.arange(len(rr_all_after_gap_mask)),
                "rr_time_sec": rr_times_all_after_gap_mask,
                "rr_interval_sec": rr_all_after_gap_mask,
                "kept_after_gap_mask": rr_gap_valid_mask
            })
            rr_gap_table.to_csv(
                os.path.join(output_folder, f"{title_base}_rr_gap_mask.csv"),
                index=False
            )

        print(f"Bad ECG segments detected from 20% RR-change rule: {len(bad_segments)}")
        print(f"Peaks after peak correction: {len(peaks_cleaned)}")
        print(f"Peaks after bad-segment removal: {len(peaks_final)}")
        print(f"Final RR intervals used for HR/HRV: {len(rr_final_cleaned)}")

        try:
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)

            if len(original_peaks) > 1:
                plot_start = max(0, original_peaks[0] - 200)
                ref_peak = original_peaks[min(len(original_peaks) - 1, 6)]
                plot_end = min(len(ecg_filtered), ref_peak + 300)
            else:
                plot_start, plot_end = choose_plot_window(ecg_filtered, start=0, end=2000, fallback=2000)

            x = np.arange(plot_start, plot_end)

            # Original peaks panel
            axes[0].plot(x - plot_start, ecg_filtered[plot_start:plot_end], color='blue', lw=1.0, label='Filtered ECG')
            orig_win = original_peaks[(original_peaks >= plot_start) & (original_peaks < plot_end)]
            if len(orig_win) > 0:
                axes[0].scatter(orig_win - plot_start, ecg_filtered[orig_win], color='red', s=32, label='Original R-peaks', zorder=3)
            axes[0].set_title(simple_title(subject, session, task, "Original R-Peaks Before signal_fixpeaks"))
            axes[0].set_ylabel("Voltage [mV]")
            axes[0].legend(loc='upper right')
            axes[0].grid(True, alpha=0.3)

            # Cleaned peaks panel
            axes[1].plot(x - plot_start, ecg_filtered[plot_start:plot_end], color='blue', lw=1.0, label='Filtered ECG')
            clean_win = peaks_cleaned[(peaks_cleaned >= plot_start) & (peaks_cleaned < plot_end)]
            if len(clean_win) > 0:
                axes[1].scatter(clean_win - plot_start, ecg_filtered[clean_win], color='green', s=32, label='Cleaned R-peaks', zorder=3)
            axes[1].set_title(simple_title(subject, session, task, "Cleaned R-Peaks After signal_fixpeaks"))
            axes[1].set_ylabel("Voltage [mV]")
            axes[1].legend(loc='upper right')
            axes[1].grid(True, alpha=0.3)

            # RR comparison panel
            rr_original = np.diff(original_peaks) / sampling_rate if len(original_peaks) > 1 else np.array([])
            rr_clean = rr_raw

            if len(rr_original) > 0:
                axes[2].plot(rr_original, marker='o', markersize=4, linestyle='-', alpha=0.5, label='Original RR')
            if len(rr_clean) > 0:
                axes[2].plot(rr_clean, marker='o', markersize=4, linestyle='-', alpha=0.7, label='RR after signal_fixpeaks')
                if len(invalid_indices) > 0:
                    axes[2].scatter(
                        invalid_indices,
                        rr_clean[invalid_indices],
                        color='red',
                        marker='x',
                        s=45,
                        label='Flagged by 20% change rule',
                        zorder=4
                    )
            axes[2].set_title(simple_title(subject, session, task, "RR Before/After signal_fixpeaks (0.30-0.75 s) + 20% Change Rule"))
            axes[2].set_xlabel("Beat Number")
            axes[2].set_ylabel("RR Interval (s)")
            axes[2].set_ylim(0, 2)
            axes[2].legend(loc='upper right')
            axes[2].grid(True, alpha=0.3)

            fig.suptitle(simple_title(subject, session, task, "signal_fixpeaks Diagnostic"), fontsize=13, fontweight='bold')
            finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_signal_fixpeaks.png"))

        except Exception as e:
            save_placeholder_plot(
                os.path.join(output_folder, f"{title_base}_signal_fixpeaks.png"),
                simple_title(subject, session, task, "Signal FixPeaks"),
                f"signal_fixpeaks ran, but the custom diagnostic plot failed: {e}"
            )

        # 10. Original vs cleaned peak comparison 
        plot_peak_correction_comparison(
            ecg_filtered, original_peaks, peaks_cleaned, sampling_rate,
            title_base, output_folder, task_label
        )

        # 11. Cleaned RR intervals
       
        rr_cleaned = rr_final_cleaned

        fig, ax = plt.subplots(figsize=WIDE_FIG, constrained_layout=True)

        if len(rr_raw) > 0:
            # Plot RR after signal_fixpeaks, but do not connect intervals removed by the 20% rule.
            rr_20pct_plot = rr_raw.astype(float).copy()
            rr_20pct_plot[~rr_valid_mask_20pct] = np.nan
            ax.plot(
                np.arange(len(rr_20pct_plot)),
                rr_20pct_plot,
                marker='o',
                linestyle='-',
                label='RR after signal_fixpeaks, 20% rule applied',
                markersize=4,
                alpha=0.6
            )

            if len(invalid_indices) > 0:
                ax.scatter(
                    invalid_indices,
                    rr_raw[invalid_indices],
                    color='red',
                    marker='x',
                    s=45,
                    label='Removed by 20% change rule',
                    zorder=4
                )

        if len(rr_final_cleaned) > 0:
            ax.plot(
                np.arange(len(rr_final_cleaned)),
                rr_final_cleaned,
                marker='o',
                linestyle='None',
                color='green',
                label='Final RR used for HR/HRV',
                markersize=4
            )

        ax.set_title(simple_title(subject, session, task, "Final Cleaned RR Intervals"))
        ax.set_xlabel("RR Interval Index")
        ax.set_ylabel("RR Interval (s)")
        ax.set_ylim(0, 2)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_cleaned_rr_intervals.png"))

        # 12. Last 10 beats
        if len(peaks_final) >= 2:
            last10_peaks = peaks_final[-min(10, len(peaks_final)):]
            start_idx = max(0, last10_peaks[0] - 200)
            end_idx = min(len(ecg_filtered), last10_peaks[-1] + 200)
            ecg_window = ecg_filtered[start_idx:end_idx]
            peaks_in_window = last10_peaks - start_idx

            fig, ax = plt.subplots(figsize=WIDE_FIG, constrained_layout=True)
            ax.plot(ecg_window, color='red', label='Filtered ECG')
            ax.plot(peaks_in_window, ecg_window[peaks_in_window], 'bo', label='R-peaks', markersize=6)
            ax.set_title(simple_title(subject, session, task, "Filtered ECG - Last 10 R-Peaks"))
            ax.set_xlabel("Samples [N]")
            ax.set_ylabel("Amplitude [mV]")
            ax.legend(loc='lower right')   # requested
            ax.grid(True, alpha=0.3)
            finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_last10_beats.png"))
        else:
            save_placeholder_plot(
                os.path.join(output_folder, f"{title_base}_last10_beats.png"),
                simple_title(subject, session, task, "Filtered ECG - Last 10 R-Peaks"),
                "Not enough peaks to plot last 10 beats."
            )

        # 13. EDR
        try:
            ecg_rate_edr = nk.signal_rate(peaks_final, sampling_rate=sampling_rate, desired_length=len(ecg_filtered))
            methods = ['vangent2019', 'soni2019', 'charlton2016', 'sarkar2015']
            fig, axes = plt.subplots(len(methods), 1, figsize=(18, 10), sharex=True, constrained_layout=True)
            for i, method in enumerate(methods):
                edr = nk.ecg_rsp(ecg_rate_edr, sampling_rate=sampling_rate, method=method)
                axes[i].plot(edr, lw=0.8, color='blue')
                axes[i].set_title(f"EDR - {method}")
                axes[i].set_ylabel("Amplitude [mV]")
                axes[i].grid(True, alpha=0.3)
            axes[-1].set_xlabel("Samples [N]")
            fig.suptitle(simple_title(subject, session, task, "ECG-Derived Respiration (EDR)"), fontsize=13, fontweight='bold')
            finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_EDR.png"))
        except Exception as e:
            save_placeholder_plot(
                os.path.join(output_folder, f"{title_base}_EDR.png"),
                simple_title(subject, session, task, "ECG-Derived Respiration (EDR)"),
                f"Could not compute EDR: {e}"
            )

        # 14. ECG delineation
        try:
            nk.ecg_delineate(
                ecg_filtered, rpeaks=peaks_final, sampling_rate=sampling_rate,
                method='dwt', show=True, show_type='all', check=True,
                window_start=-0.2, window_end=0.2
            )
            fig = plt.gcf()
            fig.set_size_inches(18, 8)
            ax = fig.axes[0]
            ax.set_title(simple_title(subject, session, task, "ECG Delineation"))
            ax.set_xlabel("Time [Seconds]")
            ax.set_ylabel("Amplitude [mV]")
            if ax.get_legend() is not None:
                ax.legend(loc='upper right', bbox_to_anchor=(1.02, 1))
            finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_delineation.png"))
        except Exception as e:
            save_placeholder_plot(
                os.path.join(output_folder, f"{title_base}_delineation.png"),
                simple_title(subject, session, task, "ECG Delineation"),
                f"Could not compute delineation: {e}"
            )

        # 15. ECG rate

        ecg_rate = compute_safe_interpolated_hr_from_rr(
            rr_times=rr_final_times,
            rr_intervals=rr_final_cleaned,
            sampling_rate=sampling_rate,
            signal_length=len(ecg_filtered),
            bad_segments=bad_segments,
            max_gap_sec=MAX_RR_FOR_INTERPOLATION
        )

        mean_hr, min_hr, max_hr, std_hr = nan_summary(ecg_rate)

        print("\n--- Heart Rate Summary ---")
        print(f"Mean HR:  {mean_hr:.1f} bpm")
        print(f"Min HR:   {min_hr:.1f} bpm")
        print(f"Max HR:   {max_hr:.1f} bpm")
        print(f"Std HR:   {std_hr:.1f} bpm")

        fig, ax = plt.subplots(figsize=WIDE_FIG, constrained_layout=True)
        sample_time = np.arange(len(ecg_rate)) / sampling_rate
        ax.plot(sample_time, ecg_rate, color='green', lw=0.8, label='Heart Rate from final cleaned RR (gaps over bad ECG segments)')

        if not np.isnan(mean_hr):
            ax.axhline(mean_hr, color='red', linestyle='--', label=f'Mean HR: {mean_hr:.1f} bpm')

        already_labeled = False
        for seg in bad_segments:
            ax.axvspan(
                seg["start_sec"],
                seg["end_sec"],
                color="red",
                alpha=0.12,
                label="Removed bad segment" if not already_labeled else None
            )
            already_labeled = True

        ax.set_title(simple_title(subject, session, task, "ECG Heart Rate Over Time"))
        ax.set_xlabel("Task Time After bas+ (seconds)")
        ax.set_ylabel("Heart Rate (bpm)")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_heart_rate.png"))





        # -------------------------------------------------------------------------
        # INTERPOLATED RR 
        # -------------------------------------------------------------------------

        sample_times = np.arange(len(ecg_rate)) / sampling_rate

        # Build the final RR array on the original task timeline before interpolation.
        rr_plot_by_time = rr_all_after_gap_mask.astype(float).copy()
        rr_plot_by_time[~rr_gap_valid_mask] = np.nan

        # 2. Build the mask safely from the matched interval array
        valid_indices = ~np.isnan(rr_plot_by_time)
        valid_rr_times = rr_times_all_after_gap_mask[valid_indices]
        valid_rr_intervals = rr_plot_by_time[valid_indices]

        if len(valid_rr_times) > 1:
            # nk.signal_interpolate takes the old X, old Y, and the target new X grid.
            # We use method='cubic' to get a smooth physiological curve.
            rr_interpolated = nk.signal_interpolate(
                x_values=valid_rr_times,
                y_values=valid_rr_intervals,
                x_new=sample_times,
                method='cubic'
            )
            
            # Re-mask the bad segments so the line breaks visually over noise blocks
            for seg in bad_segments:
                mask_gap = (sample_times >= seg["start_sec"]) & (sample_times <= seg["end_sec"])
                rr_interpolated[mask_gap] = np.nan
        else:
            rr_interpolated = np.full_like(sample_times, np.nan)
        # -------------------------------------------------------------------------





        # 16. RR vs HR
        ## Remember ecg_rate function uses nk.signal_interpolate for RR intervals
        sample_times = np.arange(len(ecg_rate)) / sampling_rate
        fig, ax1 = plt.subplots(figsize=(16, 6), constrained_layout=True)

        if len(rr_all_after_gap_mask) > 0:
            rr_plot_by_time = rr_all_after_gap_mask.astype(float).copy()
            rr_plot_by_time[~rr_gap_valid_mask] = np.nan

            ax1.plot(
                sample_times,
                rr_interpolated,
                linestyle='-',
                linewidth=0.8,
                marker='o',
                markersize=2.5,
                markevery=max(1, int(sampling_rate * 0.25)),
                color='steelblue',
                alpha=0.85,
                label='Interpolated RR intervals (bad gaps removed)'
            )

        already_labeled = False
        for seg in bad_segments:
            ax1.axvspan(
                seg["start_sec"],
                seg["end_sec"],
                color="red",
                alpha=0.12,
                label="Removed bad segment" if not already_labeled else None
            )
            already_labeled = True

        ax1.set_xlabel("Task Time After bas+ (seconds)")
        ax1.set_ylabel("RR Interval (seconds)", color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_ylim(0, 2)

        ax2 = ax1.twinx()

        # Calculate HR from the exact same interpolated RR series plotted above.
        # This guarantees the mathematical relationship HR = 60 / RR.
        hr_from_interpolated_rr = np.full_like(
            rr_interpolated,
            np.nan,
            dtype=float
        )
        valid_rr_for_hr = np.isfinite(rr_interpolated) & (rr_interpolated > 0)
        hr_from_interpolated_rr[valid_rr_for_hr] = (
            60.0 / rr_interpolated[valid_rr_for_hr]
        )

        ax2.plot(
            sample_times,
            hr_from_interpolated_rr,
            color='crimson',
            linewidth=0.8,
            alpha=0.75,
            label='Heart Rate calculated as 60 / interpolated RR'
        )

        ax2.set_ylabel("Heart Rate (bpm)", color='crimson')
        ax2.tick_params(axis='y', labelcolor='crimson')

        # PI-requested HR axis: start at 0 bpm and use 25-bpm increments.
        finite_hr = hr_from_interpolated_rr[np.isfinite(hr_from_interpolated_rr)]
        if len(finite_hr) > 0:
            hr_axis_max = max(175, int(np.ceil(np.nanmax(finite_hr) / 25.0) * 25))
        else:
            hr_axis_max = 175

        ax2.set_ylim(0, hr_axis_max)
        ax2.set_yticks(np.arange(0, hr_axis_max + 25, 25))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax1.set_title(simple_title(subject, session, task, "Final RR Intervals vs Heart Rate"))
        ax1.grid(True, alpha=0.3)
        finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_rr_vs_hr.png"))




        # 17. ECG segmentation
        try:
            nk.ecg_segment(ecg_filtered, rpeaks=peaks_final, sampling_rate=sampling_rate, show=True)
            fig = plt.gcf()
            if figure_has_content(fig):
                fig.set_size_inches(16, 6)
                finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_segmented_heartbeat.png"))
            else:
                plt.close(fig)
                save_placeholder_plot(
                    os.path.join(output_folder, f"{title_base}_segmented_heartbeat.png"),
                    simple_title(subject, session, task, "ECG Segmentation"),
                    "ECG segmentation ran, but NeuroKit did not create a visible figure for this file."
                )
        except Exception as e:
            save_placeholder_plot(
                os.path.join(output_folder, f"{title_base}_segmented_heartbeat.png"),
                simple_title(subject, session, task, "ECG Segmentation"),
                f"Could not segment heartbeats: {e}"
            )

        plt.close('all')

        # 18. HRV default
        try:
            nk.hrv(peaks_for_hrv, sampling_rate=sampling_rate, show=True)
            fig_hrv = plt.gcf()
            if figure_has_content(fig_hrv):
                fig_hrv.set_size_inches(*HRV_FIG)
                try:
                    fig_hrv.tight_layout(rect=(0, 0, 1, 0.97))
                except Exception:
                    pass
                fig_hrv.savefig(os.path.join(output_folder, f"{title_base}_HRV_Default.png"), bbox_inches='tight', dpi=150)
                axes_hrv = fig_hrv.get_axes()
            else:
                axes_hrv = []
                plt.close(fig_hrv)
                save_placeholder_plot(
                    os.path.join(output_folder, f"{title_base}_HRV_Default.png"),
                    simple_title(subject, session, task, "HRV Default"),
                    "HRV calculation ran, but NeuroKit did not create a visible figure for this file."
                )
        except Exception as e:
            axes_hrv = []
            save_placeholder_plot(
                os.path.join(output_folder, f"{title_base}_HRV_Default.png"),
                simple_title(subject, session, task, "HRV Default"),
                f"Could not compute HRV default plot: {e}"
            )

        # 18. B. HRV Time Domain Metrics
        try:
            # Capture the results in a DataFrame
            hrv_time_df = nk.hrv_time(peaks_for_hrv, sampling_rate=sampling_rate, show=True)
            fig_hrv_time = plt.gcf()
            
            if figure_has_content(fig_hrv_time):
                # Extract the metrics
                mean_nn = hrv_time_df['HRV_MeanNN'].values[0]
                sdnn = hrv_time_df['HRV_SDNN'].values[0]
                rmssd = hrv_time_df['HRV_RMSSD'].values[0]
                median_nn = hrv_time_df['HRV_MedianNN'].values[0]
                
                # Print to console
                print(f"Subject {subject} - HRV Time Domain -> MeanNN: {mean_nn:.2f}, SDNN: {sdnn:.2f}, RMSSD: {rmssd:.2f}, MedianNN: {median_nn:.2f}")

                # Add metrics to the figure
                ax = fig_hrv_time.axes[0]
                text_str = (f"MeanNN   : {mean_nn:>8.2f} ms\n"
                            f"SDNN     : {sdnn:>8.2f} ms\n"
                            f"RMSSD    : {rmssd:>8.2f} ms\n"
                            f"MedianNN : {median_nn:>8.2f} ms")
                
                ax.text(0.98, 0.98, text_str, transform=ax.transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                fig_hrv_time.set_size_inches(*HRV_FIG)
                try:
                    fig_hrv_time.tight_layout(rect=(0, 0, 1, 0.97))
                except Exception:
                    pass
                fig_hrv_time.savefig(os.path.join(output_folder, f"{title_base}_HRV_TimeDomain.png"), bbox_inches='tight', dpi=150)
            else:
                plt.close(fig_hrv_time)
                save_placeholder_plot(
                    os.path.join(output_folder, f"{title_base}_HRV_TimeDomain.png"),
                    simple_title(subject, session, task, "HRV Time Domain Metrics"),
                    "HRV time-domain metrics ran, butdid not create a visible figure."
                )

        except Exception as e:
            plt.close('all')
            save_placeholder_plot(
                os.path.join(output_folder, f"{title_base}_HRV_TimeDomain.png"),
                simple_title(subject, session, task, "HRV Time Domain Metrics"),
                f"Could not compute HRV time-domain metrics plot: {e}"
            )


        # 19. Standalone 4 Hz PSD + modified HRV plot
        try:
            if len(rr_final_cleaned) < 4:
                raise ValueError("Not enough final clean RR intervals for 4 Hz PSD after bad-segment removal.")

            # Use final clean RR intervals.
            # Convert seconds to milliseconds because HRV PSD is traditionally in ms².
            rri_ms = rr_final_cleaned.astype(float) * 1000
            rri_time = np.cumsum(rr_final_cleaned)
            interp_rate = 4
            t_interp = np.arange(rri_time[0], rri_time[-1], 1.0 / interp_rate)
            rri_interp = np.interp(t_interp, rri_time, rri_ms)
            freqs, psd = welch(rri_interp, fs=interp_rate, nperseg=min(256, len(rri_interp)))

            bands = {
                'ULF': ((freqs < 0.003), 'purple'),
                'VLF': ((freqs >= 0.003) & (freqs < 0.04), 'steelblue'),
                'LF':  ((freqs >= 0.04) & (freqs < 0.15), 'green'),
                'HF':  ((freqs >= 0.15) & (freqs < 0.4), 'orange'),
                'VHF': ((freqs >= 0.4) & (freqs < 2.0), 'red'),
            }

            fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
            for label, (mask, colour) in bands.items():
                if np.any(mask):
                    ax.fill_between(freqs[mask], psd[mask], color=colour, alpha=0.8, label=label)
            ax.set_xlim(0, 2.0)
            ax.set_ylim(bottom=0)
            ax.set_title(simple_title(subject, session, task, "4 Hz Resampled PSD"))
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Spectrum (ms²/Hz)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_standalone_4hz_psd.png"))


            try:
                nk.hrv(peaks_for_hrv, sampling_rate=sampling_rate, show=True)
                fig_hrv_new = plt.gcf()

                if figure_has_content(fig_hrv_new) and len(fig_hrv_new.get_axes()) > 1:
                    fig_hrv_new.set_size_inches(*HRV_FIG)
                    axes_new = fig_hrv_new.get_axes()

                    # In NeuroKit's HRV figure, the PSD panel is usually axes[1].
                    # This preserves the RR interval distribution and Poincaré plot.
                    ax_psd = axes_new[1]
                    ax_psd.cla()

                    for label, (mask, colour) in bands.items():
                        if np.any(mask):
                            ax_psd.fill_between(
                                freqs[mask],
                                psd[mask],
                                color=colour,
                                alpha=0.8,
                                label=label
                            )

                    ax_psd.set_xlim(0, 2.0)
                    ax_psd.set_ylim(bottom=0)
                    ax_psd.set_title('Power Spectral Density (PSD) for Frequency Domains')
                    ax_psd.set_xlabel('Frequency (Hz)')
                    ax_psd.set_ylabel('Spectrum (ms²/Hz)')
                    ax_psd.legend(loc='upper right', fontsize=8)
                    ax_psd.grid(True, alpha=0.3)

                    try:
                        fig_hrv_new.tight_layout(rect=(0, 0, 1, 0.97))
                    except Exception:
                        pass

                    fig_hrv_new.savefig(
                        os.path.join(output_folder, f"{title_base}_HRV_new_metrics.png"),
                        bbox_inches='tight',
                        dpi=150
                    )
                    plt.close(fig_hrv_new)

                else:
                    plt.close(fig_hrv_new)
                    save_placeholder_plot(
                        os.path.join(output_folder, f"{title_base}_HRV_new_metrics.png"),
                        simple_title(subject, session, task, "HRV New Metrics"),
                        "NeuroKit HRV figure did not contain the expected axes, so the combined HRV plot could not be created."
                    )

            except Exception as e:
                save_placeholder_plot(
                    os.path.join(output_folder, f"{title_base}_HRV_new_metrics.png"),
                    simple_title(subject, session, task, "HRV New Metrics"),
                    f"Could not create full HRV new metrics plot: {e}"
                )

        except Exception as e:
            save_placeholder_plot(
                os.path.join(output_folder, f"{title_base}_standalone_4hz_psd.png"),
                simple_title(subject, session, task, "4 Hz Resampled PSD"),
                f"Could not compute standalone 4 Hz PSD: {e}"
            )
            save_placeholder_plot(
                os.path.join(output_folder, f"{title_base}_HRV_new_metrics.png"),
                simple_title(subject, session, task, "HRV New Metrics"),
                f"Could not compute modified HRV plot: {e}"
            )
        






        ###################################################################################
        # 4b. Session-Wide Filtered ECG (Masked Baseline to TRSP Window)
        # ------------------------------------------------------------
        try:
            # First, run the filter on the full recording length so the signal characteristics match
            ecg_filtered_full = nk.signal_filter(
                ecg_raw_full,
                sampling_rate=sampling_rate,
                lowcut=1,
                highcut=40,
                method='butterworth',
                order=2,
                powerline=60
            )

            # Create a copy to explicitly zero out/flatten the edges
            ecg_window_masked = ecg_filtered_full.copy()
            
            # Set everything before bas+ and after TRSP to exactly 0 (or np.nan if you want gaps)
            ecg_window_masked[:task_start_sample] = 0
            ecg_window_masked[task_end_sample:] = 0

            # Establish the time vector relative to the absolute recording start
            full_recording_times = np.arange(len(ecg_filtered_full)) / sampling_rate

            fig, ax = plt.subplots(figsize=WIDE_FIG, constrained_layout=True)
            
            # Plot the masked signal across the entire chronological timeline
            ax.plot(full_recording_times, ecg_window_masked, color='blue', lw=0.8, 
                    label='Isolated Experimental Window (bas+ to TRSP)')
            
            # Add visual boundary markers for clarity

            if 'bgin_marker_sec' in locals() and bgin_marker_sec is not None:
                ax.axvline(
                    bgin_marker_sec,
                    color='black',
                    linestyle=':',
                    linewidth=1.5,
                    label=f'bgin ({bgin_marker_sec:.2f}s)',
                    zorder=4
                )

            if 'din3_marker_sec' in locals() and din3_marker_sec is not None:
                ax.axvline(
                    din3_marker_sec,
                    color='gold',
                    linestyle='-',
                    linewidth=1.0,
                    alpha=0.8,
                    label=f'DIN3 ({din3_marker_sec:.2f}s)',
                    zorder=6
                )
            
            if marker_start_sec is not None:
                ax.axvline(
                    marker_start_sec,
                    color='red',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.95,
                    label=f'bas+ ({marker_start_sec:.2f}s)',
                    zorder=5
                )
            if marker_end_sec is not None:
                ax.axvline(marker_end_sec, color='black', linestyle='-.', alpha=0.7, 
                           label=f'TRSP ({marker_end_sec:.2f}s)')

            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.set_title(simple_title(subject, session, task, "Isolation Window (Zero Arrays Added) Simulated Full Filtered ECG Signal"))
            ax.set_xlabel("Absolute Session Time (seconds)")
            ax.set_ylabel("Amplitude [mV]")
            ax.set_xlim(0, full_duration_sec)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            finalize_and_save(fig, os.path.join(output_folder, f"{title_base}_filtered_task_window_isolated.png"))



        # PLOT 2: Zoomed-In View (Beginning Timeline Focus) ---
            # ------------------------------------------------------------
            fig_zoom, ax_zoom = plt.subplots(figsize=WIDE_FIG, constrained_layout=True)
            
            # Plot the same underlying data
            ax_zoom.plot(full_recording_times, ecg_window_masked, color='blue', lw=1.0, 
                         label='Isolated Experimental Window')
            
            # Re-draw the vertical markers
            if 'bgin_marker_sec' in locals() and bgin_marker_sec is not None:
                ax_zoom.axvline(bgin_marker_sec, color='black', linestyle=':', linewidth=2.0, alpha=0.9,
                                label=f'bgin ({bgin_marker_sec:.2f}s)')

            if 'din3_marker_sec' in locals() and din3_marker_sec is not None:
                ax_zoom.axvline(
                    din3_marker_sec,
                    color='gold',
                    linestyle='-',
                    linewidth=1.5,
                    alpha=0.95,
                    label=f'DIN3 ({din3_marker_sec:.2f}s)',
                    zorder=6
                )
            if marker_start_sec is not None:
                ax_zoom.axvline(
                    marker_start_sec,
                    color='red',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.95,
                    label=f'bas+ ({marker_start_sec:.2f}s)',
                    zorder=5
                )

            # Focus the X-axis strictly from 3 seconds before the start marker and 5 seconds past the start marker
            zoom_start_x = marker_start_sec - 3 if marker_start_sec is not None else 0.0
            zoom_end_x = (marker_start_sec + 5) if marker_start_sec is not None else 30.0
            ax_zoom.set_xlim(zoom_start_x, zoom_end_x)
            
            # Dynamically adjust Y-limits to fit only the signal data visible inside this zoom window
            visible_indices = (full_recording_times >= zoom_start_x) & (full_recording_times <= zoom_end_x)
            if np.any(visible_indices):
                visible_data = ecg_window_masked[visible_indices]
                v_min, v_max = np.min(visible_data), np.max(visible_data)
                padding = 0.1 * (v_max - v_min) if v_max != v_min else 0.01
                ax_zoom.set_ylim(v_min - padding, v_max + padding)


            # Add marker labels inside the zoom plot.
            if np.any(visible_indices):
                label_y = v_max + padding * 0.85

                if 'bgin_marker_sec' in locals() and bgin_marker_sec is not None:
                    ax_zoom.text(
                        bgin_marker_sec - 0.10,
                        label_y,
                        'bgin',
                        color='black',
                        rotation=90,
                        ha='right',
                        va='top',
                        fontweight='bold',
                        fontsize=10
                    )

                if marker_start_sec is not None:
                    ax_zoom.text(
                        marker_start_sec - 0.03,
                        label_y,
                        'bas+',
                        color='red',
                        rotation=90,
                        ha='right',
                        va='top',
                        fontweight='bold',
                        fontsize=10
                    )

                if 'din3_marker_sec' in locals() and din3_marker_sec is not None:
                    ax_zoom.text(
                        din3_marker_sec + 0.03,
                        label_y,
                        'DIN3',
                        color='goldenrod',
                        rotation=90,
                        ha='left',
                        va='top',
                        fontweight='bold',
                        fontsize=10
                    )


            ax_zoom.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax_zoom.set_title(simple_title(subject, session, task, "Task Isolation Window - Zoomed Beginning (bgin / bas+ / DIN3)"))
            ax_zoom.set_xlabel("Absolute Session Time (seconds)")
            ax_zoom.set_ylabel("Amplitude [mV]")
            ax_zoom.grid(True, alpha=0.3)
            ax_zoom.legend(loc='upper left')
            
            finalize_and_save(fig_zoom, os.path.join(output_folder, f"{title_base}_filtered_task_window_isolated_zoom.png"))
            
        except Exception as e:
            save_placeholder_plot(
                os.path.join(output_folder, f"{title_base}_filtered_task_window_isolated.png"),
                simple_title(subject, session, task, "Filtered ECG - Isolated Window"),
                f"Could not compute or plot session-wide isolated task window: {e}"
            )

        ########################################################################
        ########################################################################

        # Total duration of the analyzed task window.
        # bad_segments are task-relative, so the denominator should use the task duration.
        total_duration = len(ecg_raw) / sampling_rate

        # Sum up the exact duration of the segments 
        total_red_gaps_sec = sum(seg["end_sec"] - seg["start_sec"] for seg in bad_segments)

        # Calculate the Usable Epoch Percentage 
        if total_duration > 0:
            usable_epoch_pct = ((total_duration - total_red_gaps_sec) / total_duration) * 100
        else:
            usable_epoch_pct = 0.0

        print(f"Usable Epoch: {usable_epoch_pct:.2f}%")
        ##########################################################################







        plt.close('all')

        # 20. Combined Summary Report Figure
        n_data_rows = 3 + 8 + 3 + 2 + len(onsets) + 1
        fig_height = 1.0 + n_data_rows * 0.4
        fig, ax = plt.subplots(figsize=(TABLE_WIDTH, fig_height))
        ax.axis('off')

        if high_quality_pct > 95:
            status = "Excellent - Signal is highly reliable for analysis."
        elif high_quality_pct > 80:
            status = "Good - Majority of signal is usable."
        else:
            status = "Caution - Significant noise detected."

        rows = []
        rows.append(["RECORDING INFO", "", "#2c7bb6", True])
        rows.append(["Task Samples", f"{len(ecg_raw):,}", "#f0f4f8", False])
        rows.append(["Task Duration", f"{len(ecg_raw)/sampling_rate:.1f}s  /  {len(ecg_raw)/sampling_rate/60:.2f} min", "#f0f4f8", False])

        rows.append(["QUALITY REPORT", "", "#2c7bb6", True])
        rows.append(["Absolute Minimum Quality", f"{np.min(quality):.4f}", "#ffffff", False])
        rows.append(["Absolute Maximum Quality", f"{np.max(quality):.4f}", "#ffffff", False])
        rows.append(["Mean Quality", f"{np.mean(quality):.4f}", "#ffffff", False])
        rows.append(["Median Quality", f"{np.median(quality):.4f}", "#ffffff", False])
        rows.append(["Standard Deviation", f"{np.std(quality):.4f}", "#ffffff", False])
        rows.append(["5 Lowest Indices", str(worst_indices), "#ffffff", False])
        rows.append(["5 Lowest Values", str(np.round(worst_values, 4)), "#ffffff", False])

        rows.append(["FILE RELIABILITY", "", "#2c7bb6", True])
        rows.append([f"% of file > {QUALITY_THRESHOLD} quality", f"{high_quality_pct:.2f}%", "#d9ead3", False])
        ### ADD NEW EPOCH ####
        rows.append(["Usable Epoch (%)", f"{usable_epoch_pct:.2f}%", "#d9ead3", False])
        ##########################
        rows.append(["Status", status, "#d9ead3", True])

        

        rows.append(["AUTOMATED SEGMENT DETECTION", "", "#2c7bb6", True])
        rows.append(["Total Segments Detected", str(len(onsets)), "#f0f4f8", False])
        for i, (seg_start, seg_end) in enumerate(zip(onsets, offsets)):
            start_sec_seg = seg_start / sampling_rate
            end_sec_seg = seg_end / sampling_rate
            duration_seg = (seg_end - seg_start) / sampling_rate
            bg = "#ffffff" if i % 2 == 0 else "#f7f7f7"
            rows.append([
                f"Segment {i+1}",
                f"Onset: {seg_start} ({int(start_sec_seg//60)}m {int(start_sec_seg%60)}s)  |  "
                f"Offset: {seg_end} ({int(end_sec_seg//60)}m {int(end_sec_seg%60)}s)  |  "
                f"Duration: {duration_seg:.2f}s",
                bg, False
            ])
        rows.append(["TOTAL USABLE YIELD", f"{total_usable_seconds:.2f}s  /  {total_usable_seconds/60:.2f} min", "#d9ead3", True])

        cell_text = [[r[0], r[1]] for r in rows]
        cell_colors = [[r[2], r[2]] for r in rows]
        table = ax.table(
            cellText=cell_text,
            colLabels=["Metric / Section", "Value"],
            cellColours=cell_colors,
            loc='center',
            cellLoc='left',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.auto_set_column_width(col=[0, 1])
        for (row, col), cell in table.get_celld().items():
            cell.set_height(1.0 / (len(rows) + 1))
            cell.PAD = 0.04
        for j in range(2):
            table[0, j].set_facecolor('#1a4a7a')
            table[0, j].set_text_props(color='white', fontweight='bold')
        for i, r in enumerate(rows):
            if r[3]:
                for j in range(2):
                    cell = table[i + 1, j]
                    cell.set_text_props(fontweight='bold', color='white' if r[2] == '#2c7bb6' else 'black')

        ax.set_title(simple_title(subject, session, task, "Full Signal & Quality Report"), fontweight='bold', fontsize=13, pad=12)
        fig.savefig(os.path.join(output_folder, f'{title_base}_Quality_Report.png'), bbox_inches='tight', dpi=150)
        plt.close(fig)

        # 21. RS summary CSV (metric/value format for Excel)
        summary_rows = [
            ('subject_id', subject.replace('sub-', '')),
            ('session_id', session.replace('ses-', '')),
            ('task_name', task),
            ('sampling_rate_hz', sampling_rate),
            ('full_recording_duration_seconds', round(full_duration_sec, 6)),
            ('analysis_start_sec', round(float(task_start_sec), 6)),
            ('analysis_end_sec', round(float(task_end_sec), 6)),
            ('duration_seconds', round(len(ecg_raw) / sampling_rate, 6)),
            ('duration_minutes', round(len(ecg_raw) / sampling_rate / 60, 6)),
            ('quality_mean', round(float(np.mean(quality)), 6)),
            ('quality_min', round(float(np.min(quality)), 6)),
            ('quality_max', round(float(np.max(quality)), 6)),
            ('quality_std', round(float(np.std(quality)), 6)),
            ('quality_median', round(float(np.median(quality)), 6)),
            ('high_quality_pct', round(float(high_quality_pct), 6)),
            ###### ADD NEW EPOCH METRIC ######
            ('usable_epoch_pct', round(float(usable_epoch_pct), 6)),
            ##################################
            ('num_peaks_detected', int(len(original_peaks))),
            ('fixpeaks_interval_min_sec', FIXPEAKS_INTERVAL_MIN),
            ('fixpeaks_interval_max_sec', FIXPEAKS_INTERVAL_MAX),
            ('num_peaks_cleaned_after_fixpeaks', int(len(peaks_cleaned))),
            ('num_peaks_final_after_bad_segment_removal', int(len(peaks_final))),
            ('rr_change_threshold_percent', round(RR_PERCENT_CHANGE_THRESHOLD * 100, 2)),
            ('rr_change_formula', 'abs(RR_n - RR_n_minus_1) / RR_n'),
            ('num_bad_segments_20percent', int(len(bad_segments))),
            ('rr_20percent_intervals_removed', int(beats_removed)),
            ('rr_20percent_filter_yield_pct', round(float(filter_yield_pct), 6)),
            ('num_segments', int(len(onsets))),
            ('total_usable_seconds', round(float(total_usable_seconds), 6)),
            ('mean_rr_interval_sec', round(float(np.mean(rr_cleaned)) if len(rr_cleaned) > 0 else np.nan, 6)),
            ('min_rr_interval_sec', round(float(np.min(rr_cleaned)) if len(rr_cleaned) > 0 else np.nan, 6)),
            ('max_rr_interval_sec', round(float(np.max(rr_cleaned)) if len(rr_cleaned) > 0 else np.nan, 6)),
            ('mean_heart_rate_bpm', round(float(mean_hr), 6) if not np.isnan(mean_hr) else np.nan),
            ('min_heart_rate_bpm', round(float(min_hr), 6) if not np.isnan(min_hr) else np.nan),
            ('max_heart_rate_bpm', round(float(max_hr), 6) if not np.isnan(max_hr) else np.nan),
            ('std_heart_rate_bpm', round(float(std_hr), 6) if not np.isnan(std_hr) else np.nan),
            ('start_marker_label', marker_start_label if marker_start_label is not None else ''),
            ('start_marker_sec', round(float(marker_start_sec), 6) if marker_start_sec is not None else np.nan),
            ('end_marker_label', marker_end_label if marker_end_label is not None else ''),
            ('end_marker_sec', round(float(marker_end_sec), 6) if marker_end_sec is not None else np.nan),
        ]
        save_rs_summary_csv(output_folder, title_base, summary_rows)

        print(f"\n Finished: {filename}")
        print(f"Outputs saved in: {output_folder}")

    except Exception as e:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_traceback = traceback.format_exc()
        print(f"\n CRITICAL ERROR SKIPPED FOR {filename}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"{'-'*40}")

        with open(os.path.join(OUTPUT_DIR, "failed_files_log.txt"), "a") as log:
            log.write(f"{'='*80}\n")
            log.write(f"TIMESTAMP: {current_time}\n")
            log.write(f"FILE: {filename}\n")
            log.write(f"ERROR TYPE: {type(e).__name__}\n")
            log.write(f"ERROR MESSAGE: {str(e)}\n")
            log.write(f"\n--- FULL TRACEBACK STACK ---\n")
            log.write(f"{full_traceback}")
            log.write(f"{'='*80}\n\n")

print("\nPipeline finished.")
print(f"Base output folder: {OUTPUT_DIR}")