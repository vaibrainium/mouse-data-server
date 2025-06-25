import ast
import os
import pickle
import sys
import logging
from pathlib import Path

import dotenv
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from utils import pmf_utils

# Set up logging
logging.basicConfig(level=logging.INFO)
# set logging directory to log folder
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()
shared_data_dir = os.getenv("SHARED_DATA_DIR")
processed_data_dir = os.getenv("PROCESSED_DATA_DIR")

if not shared_data_dir or not processed_data_dir:
    raise ValueError("Environment variables SHARED_DATA_DIR and PROCESSED_DATA_DIR must be set.")

SHARED_DATA_DIR = Path(shared_data_dir)
PROCESSED_DATA_DIR = Path(processed_data_dir)


# ----------------- Utility Functions -----------------

def safe_time_parse(time_str, formatted: bool = False):
    """Parse time string safely, handling None inputs."""
    if time_str is None or pd.isna(time_str):
        return None
    try:
        dt = pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce')
        if pd.isna(dt):
            return None
        return dt.strftime('%I:%M %p') if formatted else dt.time()
    except Exception:
        return None


def safe_nanmean(arr):
    """Calculate nanmean safely, returning np.nan for empty arrays."""
    try:
        if len(arr) == 0 or arr is None:
            return np.nan
        result = np.nanmean(arr)
        # Ensure we return a scalar
        return float(result) if not np.isnan(result) else np.nan
    except Exception:
        return np.nan


def logistic(x, bias, sensitivity):
    return expit(sensitivity * (x - bias))


def fit_psychometric(x, y):
    def loss(params):
        prob = logistic(x, *params)
        return -np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))
    return minimize(loss, [0, 0.1]).x  # Returns [bias, sensitivity]


def get_sensory_noise(data: pd.DataFrame):
    """Calculate sensory noise, ensuring scalar return value."""
    try:
        if data.empty:
            return np.nan
        result = pmf_utils.get_psychometric_data(data, fit=False)
        # Handle variable return length from get_psychometric_data
        if len(result) >= 2:
            coherence, choices = result[0], result[1]
        else:
            return np.nan
        _, sensitivity = fit_psychometric(coherence, choices)
        result = 1 / sensitivity if sensitivity and sensitivity != 0 else np.nan
        return float(result) if not np.isnan(result) else np.nan
    except Exception as e:
        logger.warning(f"Error calculating sensory noise: {e}")
        return np.nan


# ----------------- Data Processing Functions -----------------
def get_recent_sessions(last_X_business_days=None, start_date=None, end_date=None) -> pd.DataFrame:
    mouse_ids = [f for f in SHARED_DATA_DIR.iterdir() if f.is_dir() and f.name != "XXX"]
    today = pd.to_datetime("today")

    if last_X_business_days:
        start_date, end_date = today - pd.offsets.BDay(last_X_business_days), today
    else:
        start_date = pd.to_datetime(start_date) if start_date else None
        end_date = pd.to_datetime(end_date or today)
        if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
            raise ValueError("Invalid start_date or end_date format.")

    session_data = []
    required_cols = ["date", "start_weight", "end_weight", "baseline_weight", "rig_id", "protocol", "experiment", "session", "session_uuid"]

    for mouse in mouse_ids:
        history_path = mouse / "history.csv"
        if not history_path.exists():
            continue

        try:
            raw_history = pd.read_csv(history_path, dayfirst=True)
        except Exception as e:
            print(f"Failed to read {history_path}: {e}")
            continue

        valid_rows = []
        for i, row in raw_history.iterrows():
            try:
                row_data = row.reindex(required_cols, fill_value=None)

                # Parse date
                date_parsed = pd.to_datetime(row_data["date"], errors="coerce")
                if pd.isna(date_parsed) or not (start_date <= date_parsed <= end_date):
                    continue

                # Check required values
                if pd.isna(row_data[["start_weight", "end_weight", "baseline_weight", "rig_id", "protocol", "experiment", "session", "session_uuid"]]).any():
                    continue

                if row_data["baseline_weight"] == 0:
                    continue

                # Normalize weights
                row_data["start_weight"] = float(row_data["start_weight"]) / float(row_data["baseline_weight"]) * 100
                row_data["end_weight"] = float(row_data["end_weight"]) / float(row_data["baseline_weight"]) * 100
                row_data["date"] = date_parsed
                row_data["mouse_id"] = mouse.name
                row_data["rig_id"] = row_data["rig_id"]

                valid_rows.append(row_data)
            except Exception as e:
                print(f"Skipping row {i} in {mouse.name} due to format error: {e}")
                continue

        if valid_rows:
            session_data.append(pd.DataFrame(valid_rows))

    return pd.concat(session_data, ignore_index=True) if session_data else pd.DataFrame()

import pandas as pd

def preprocess_data(data: pd.DataFrame):
    # Filter for choices of -1 and 1
    all_data = data[data["choice"].isin([-1, 1])].reset_index(drop=True)
    valid_data = data.copy()
    if "outcome" in valid_data.columns:
        valid_data = valid_data.dropna(subset=["outcome"])
    if "is_correction_trial" in valid_data.columns:
        valid_data = valid_data[~valid_data["is_correction_trial"]]
    if "in_active_bias_correction_block" in valid_data.columns:
        valid_data = valid_data[~valid_data["in_active_bias_correction_block"]]

    return valid_data, all_data

def get_active_block_vars(data: pd.DataFrame):
    if "in_active_bias_correction_block" not in data.columns:
        return data.index, None, None, None, None

    active_blocks = data["in_active_bias_correction_block"].astype(int).diff()
    starts, ends = np.where(active_blocks > 0)[0], np.where(active_blocks < 0)[0]
    if len(ends) < len(starts):
        ends = np.append(ends, len(data) - 1)

    left_s, left_e, right_s, right_e = [], [], [], []
    for s, e in zip(starts, ends):
        block = data.loc[s:e]
        mean_signed = np.nanmean(block["signed_coherence"][block["outcome"] == 1])
        (right_s if mean_signed > 0 else left_s).append(s)
        (right_e if mean_signed > 0 else left_e).append(e)

    return data.index, right_s, right_e, left_s, left_e


def get_all_rolling_bias(data: pd.DataFrame, window=20):
    rolling_bias = np.zeros(window)
    accumulated_bias, idx = [], 0

    for trial in data.itertuples():
        if getattr(trial, "in_active_bias_correction_block", False):
            rolling_bias.fill(0)
        elif not trial.is_correction_trial and trial.outcome is not None:
            rolling_bias[idx] = trial.choice
            idx = (idx + 1) % window
        accumulated_bias.append(np.mean(rolling_bias))

    return accumulated_bias

def get_psychometric_data(data: pd.DataFrame, model_type="logit_3"):
    x_data, y_data, model, x_hat, y_hat = pmf_utils.get_psychometric_data(data, model_type=model_type, lapse_rate_lims=(1e-5, 0.5))
    return {
        "x_data": x_data,
        "y_data": y_data,
        "x_hat": x_hat,
        "y_hat": y_hat,
        "coefs_": model.coefs_ if model else None,
    } if model else None



def process_analyzed_data(all_trial_info, valid_trial_info):
    coherences, accuracies = pmf_utils.get_accuracy_data(valid_trial_info)
    _, median_rt, mean_rt, std_rt = pmf_utils.get_chronometric_data(valid_trial_info)
    idx, r_starts, r_ends, l_starts, l_ends = get_active_block_vars(all_trial_info)
    rolling_bias = get_all_rolling_bias(all_trial_info)

    return {
        "binned_trials": valid_trial_info.idx_valid,
        "binned_accuracies": valid_trial_info.outcome.rolling(20).mean() * 100,
        "rolling_trials": valid_trial_info.idx_valid,
        "rolling_bias": valid_trial_info.choice.rolling(20).mean(),
        "coherences": coherences,
        "accuracy": accuracies,
        "reaction_time_mean": mean_rt,
        "reaction_time_median": median_rt,
        "valid_psych_data": get_psychometric_data(valid_trial_info),
        "all_psych_data": get_psychometric_data(all_trial_info),
        "reaction_time_sd": std_rt,
        "all_data_idx": idx,
        "all_data_rolling_bias": rolling_bias,
        "all_data_choice": all_trial_info.choice,
        "right_active_block_starts": r_starts,
        "right_active_block_ends": r_ends,
        "left_active_block_starts": l_starts,
        "left_active_block_ends": l_ends,
    }


def load_existing_data():
    session_file = PROCESSED_DATA_DIR / "session_info.csv"
    analyzed_file = PROCESSED_DATA_DIR / "analyzed_data.pkl"
    if session_file.exists() and analyzed_file.exists():
        return pd.read_csv(session_file), pickle.load(open(analyzed_file, "rb"))
    return pd.DataFrame(), {}


def update_sessions(old_info, old_data, new_info):
    to_remove = old_info[~old_info["session_uuid"].isin(new_info["session_uuid"])]["session_uuid"]
    to_add = new_info[~new_info["session_uuid"].isin(old_info["session_uuid"])]
    for uuid in to_remove:
        old_data.pop(uuid, None)
    updated_info = new_info[~new_info["session_uuid"].isin(to_remove)].reset_index(drop=True)
    return updated_info, old_data, to_add


def process_rdk_analysis(meta, new_sessions, analyzed_data, mouse_id, date, trial_path, summary_path):
        try:
            trials = pd.read_csv(trial_path)
        except Exception as e:
            print(f"Error reading trial file {trial_path}: {e}")
            new_sessions = new_sessions[~((new_sessions.mouse_id == mouse_id) & (new_sessions.session == meta.session))]
            return

        valid_trials, all_trials = preprocess_data(trials)
        if all_trials.shape[0] < 10:
            new_sessions = new_sessions[~((new_sessions.mouse_id == mouse_id) & (new_sessions.session == meta.session))]
            print(f"Not enough trials for {mouse_id} on {date}. Skipping.")
            return

        try:
            summary = pd.read_csv(summary_path, on_bad_lines='skip', quoting=1)
        except Exception as e:
            print(f"Error reading summary file {summary_path}: {e}")
            try:
                # Try with different parsing options
                summary = pd.read_csv(summary_path, on_bad_lines='skip', sep=',', quotechar='"', skipinitialspace=True)
            except Exception as e2:
                print(f"Failed to read summary file with alternative parsing: {e2}")
                new_sessions = new_sessions[~((new_sessions.mouse_id == mouse_id) & (new_sessions.session == meta.session))]
                return
        summary_row = summary[summary.session_uuid == meta.session_uuid]

        start_time_raw = summary_row.start_time.values[0] if not summary_row.empty else None
        end_time_raw = summary_row.end_time.values[0] if not summary_row.empty else None

        condition = (new_sessions.mouse_id == mouse_id) & (new_sessions.date == date) & (new_sessions.session == meta.session)
        # Extract scalar values to avoid inhomogeneous array issues
        configuration_used = summary_row.configuration_used.values[0] if "configuration_used" in summary_row.columns and not summary_row.empty else None
        comments = summary_row.comments.values[0] if "comments" in summary_row.columns and not summary_row.empty else None

        # Ensure scalar values for numeric columns
        total_attempts = float(max(valid_trials.idx_attempt)) if not valid_trials.idx_attempt.empty else np.nan
        total_valid = float(max(valid_trials.idx_valid)) if not valid_trials.idx_valid.empty else np.nan

        # Calculate accuracy metrics ensuring scalar return
        session_accuracy = safe_nanmean(valid_trials.outcome) * 100 if not valid_trials.empty else np.nan
        left_mask = np.sign(valid_trials.signed_coherence) == -1 if not valid_trials.empty else pd.Series([], dtype=bool)
        right_mask = np.sign(valid_trials.signed_coherence) == 1 if not valid_trials.empty else pd.Series([], dtype=bool)
        left_accuracy = safe_nanmean(valid_trials.outcome[left_mask]) * 100 if not valid_trials.empty and left_mask.any() else np.nan
        right_accuracy = safe_nanmean(valid_trials.outcome[right_mask]) * 100 if not valid_trials.empty and right_mask.any() else np.nan

        # Calculate other metrics ensuring scalar types
        total_reward = int(np.nansum(valid_trials.trial_reward)) if not valid_trials.empty and 'trial_reward' in valid_trials.columns else 0
        sensory_noise = get_sensory_noise(all_trials) if not all_trials.empty else np.nan

        # Prepare values list ensuring all are scalars
        values = [
            safe_time_parse(start_time_raw),
            safe_time_parse(end_time_raw),
            configuration_used,
            total_attempts,
            total_valid,
            session_accuracy,
            left_accuracy,
            right_accuracy,
            total_reward,
            sensory_noise,
            comments,
        ]

        # Verify all values are scalars before assignment
        for i, val in enumerate(values):
            if hasattr(val, '__len__') and not isinstance(val, str) and val is not None:
                logger.warning(f"Non-scalar value detected at index {i}: {val}")
                values[i] = np.nan if isinstance(val, (list, tuple, np.ndarray)) else val

        new_sessions.loc[condition, [
            "start_time", "end_time", "configuration_used", "total_attempts", "total_valid", "session_accuracy",
            "left_accuracy", "right_accuracy", "total_reward", "sensory_noise", "comments"
        ]] = values

        analyzed_data[meta.session_uuid] = process_analyzed_data(all_trials, valid_trials)

def process_basic_analysis(meta, new_sessions, analyzed_data, mouse_id, date, trial_path, summary_path):
    try:
        all_trials = pd.read_csv(trial_path)
    except Exception as e:
        print(f"Error reading trial file {trial_path}: {e}")
        return

    try:
        summary = pd.read_csv(summary_path, on_bad_lines='skip', quoting=1)
    except Exception as e:
        print(f"Error reading summary file {summary_path}: {e}")
        try:
            # Try with different parsing options
            summary = pd.read_csv(summary_path, on_bad_lines='skip', sep=',', quotechar='"', skipinitialspace=True)
        except Exception as e2:
            print(f"Failed to read summary file with alternative parsing: {e2}")
            return
    summary_row = summary[summary.session_uuid == meta.session_uuid]

    start_time_raw = summary_row.start_time.values[0] if not summary_row.empty else None
    end_time_raw = summary_row.end_time.values[0] if not summary_row.empty else None

    condition = (new_sessions.mouse_id == mouse_id) & (new_sessions.date == date) & (new_sessions.session == meta.session)

    # Extract scalar values to avoid inhomogeneous array issues
    configuration_used = summary_row.configuration_used.values[0] if "configuration_used" in summary_row.columns and not summary_row.empty else None
    comments = summary_row.comments.values[0] if "comments" in summary_row.columns and not summary_row.empty else None
    total_attempts = float(max(all_trials.idx_attempt)) if not all_trials.idx_attempt.empty else np.nan
    total_reward = int(np.nansum(all_trials.trial_reward)) if 'trial_reward' in all_trials.columns else 0

    new_sessions.loc[condition, ["start_time", "end_time", "configuration_used", "total_attempts", "total_reward", "comments", "session_accuracy", "sensory_noise", "total_valid"]] = [
        safe_time_parse(start_time_raw),
        safe_time_parse(end_time_raw),
        configuration_used,
        total_attempts,
        total_reward,
        comments,
        np.nan,
        np.nan,
        total_attempts
    ]

    analyzed_data[meta.session_uuid] =  {
        "all_data_idx": all_trials.index,
        "all_data_rolling_bias": all_trials.choice.rolling(20).mean(),
        "all_data_choice": all_trials.choice,

    }

# ----------------- Main Execution -----------------

if __name__ == "__main__":
    old_info, old_data = load_existing_data()
    session_info = get_recent_sessions(last_X_business_days=30)
    session_info["date"] = pd.to_datetime(session_info.date).dt.date

    if session_info.empty:
        print("No session data found. Exiting.")
        exit()

    if old_info.empty:
        new_sessions, analyzed_data = session_info, {}
    else:
        session_info, analyzed_data, new_sessions = update_sessions(old_info, old_data, session_info)
        if new_sessions.empty:
            print("No new sessions to process.")
            exit()

    for mouse_id in new_sessions.mouse_id.unique():
        mouse_sessions = new_sessions[new_sessions.mouse_id == mouse_id]
        for date in mouse_sessions.date.unique():
            sessions = mouse_sessions[mouse_sessions.date == date].reset_index()
            for _, meta in sessions.iterrows():
                trial_path = SHARED_DATA_DIR / mouse_id / "data" / meta.protocol / meta.experiment / meta.session / f"{mouse_id}_trial.csv"
                summary_path = SHARED_DATA_DIR / mouse_id / "data" / meta.protocol / meta.experiment / f"{mouse_id}_summary.csv"

                if not trial_path.exists() or not summary_path.exists():
                    print(f"Missing files for {mouse_id} on {date}. Skipping.")
                    continue

                if meta.experiment in ["rt_directional_training", "rt_maintenance", "rt_test", "rt_dynamic_training"]:
                    process_rdk_analysis(meta, new_sessions, analyzed_data, mouse_id, date, trial_path, summary_path)
                elif meta.experiment in ["free_reward_training", "reward_spout_association", "reward_spout_stimulus_association"]:
                    process_basic_analysis(meta, new_sessions, analyzed_data, mouse_id, date, trial_path, summary_path)
                else:
                    print(f"Unknown experiment type '{meta.experiment}' for {meta.mouse_id} on {meta.date}. Skipping.")


    # Save outputs
    updated_info = pd.concat([old_info, new_sessions], ignore_index=True).drop_duplicates("session_uuid", keep="last")
    updated_info.to_csv(PROCESSED_DATA_DIR / "session_info.csv", index=False)
    with open(PROCESSED_DATA_DIR / "analyzed_data.pkl", "wb") as f:
        pickle.dump(analyzed_data, f)

    print(f"Processing complete. {len(new_sessions)} new sessions added.")
