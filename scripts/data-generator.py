import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import dotenv
import os

import scipy.optimize as opt
from scipy.stats import norm
from scipy.special import expit  # Sigmoid function

from utils import pmf_utils
dotenv.load_dotenv()

SHARED_DATA_DIR = Path(os.getenv("SHARED_DATA_DIR"))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR"))


def get_recent_sessions(last_X_business_days=None, start_date=None, end_date=None):

    # Get all mouse ID folders except "XXX"
    mouse_ids = [f for f in SHARED_DATA_DIR.iterdir() if f.is_dir() and f.name != "XXX"]

    # Determine date range
    today = pd.to_datetime("today")  # Keep full timestamp
    if last_X_business_days is not None:
        start_date = today - pd.offsets.BDay(last_X_business_days)
        end_date = today
    else:
        start_date = pd.to_datetime(start_date, errors="coerce")
        end_date = pd.to_datetime(end_date, errors="coerce") if end_date else today

    # Validate date range
    if pd.isna(start_date) or pd.isna(end_date):
        raise ValueError("Invalid start_date or end_date format.")

    # Convert start_date and end_date to a consistent format
    date_format = "%Y-%m-%d"
    start_date_str = start_date.strftime(date_format)
    end_date_str = end_date.strftime(date_format)

    session_data = []
    required_columns = ["date", "start_weight", "end_weight", "baseline_weight", "experiment", "session", "session_uuid"]

    for mouse in mouse_ids:
        history_path = mouse / "history.csv"

        try:
            history = pd.read_csv(history_path)
        except FileNotFoundError:
            continue

        # Convert date column dynamically to handle mixed formats
        history["date"] = pd.to_datetime(history["date"], format="mixed", errors="coerce")
        history = history.dropna(subset=["date"])  # Drop rows with invalid dates

        # Reformat history["date"] to match start_date and end_date format
        history["date"] = history["date"].dt.strftime(date_format)
        history["date"] = pd.to_datetime(history["date"], format=date_format, errors="coerce")  # Convert back to datetime

        # Apply date filtering
        history = history[(history["date"] >= start_date) & (history["date"] <= end_date)]

        if history.empty:
            continue

        # Ensure required columns exist
        history = history.reindex(columns=required_columns, fill_value=pd.NA)
        history["start_weight"] = history["start_weight"] / history["baseline_weight"] * 100
        history["end_weight"] = history["end_weight"] / history["baseline_weight"] * 100
        history["mouse_id"] = mouse.name

        session_data.append(history[["mouse_id", "date", "start_weight", "end_weight", "experiment", "session", "session_uuid"]])

    # Concatenate all results into a single DataFrame
    return pd.concat(session_data, ignore_index=True) if session_data else pd.DataFrame()

def preprocess_data(data):
    all_data =data[data["choice"].isin([-1, 1])].reset_index(drop=True)

    valid_data = data.dropna(subset=["outcome"])
    valid_data = valid_data[valid_data.is_correction_trial == False]
    if "in_active_bias_correction_block" in valid_data.columns:
        valid_data = valid_data[valid_data.in_active_bias_correction_block == False]
    return valid_data, all_data

def get_binned_accuracy(data, bin_size=20):
    outcome_array = data["outcome"].astype(float).to_numpy()
    num_bins = len(outcome_array) // bin_size
    binned_accuracy = [np.nanmean(outcome_array[i * bin_size : (i + 1) * bin_size]) * 100 for i in range(num_bins)]
    binned_indices = np.arange(num_bins) * bin_size
    return binned_indices, np.array(binned_accuracy)

def get_active_block_vars(data):
    if "in_active_bias_correction_block" not in data.columns:
        return data.index, None, None, None, None

    active_block_starts = np.where(data["in_active_bias_correction_block"].astype(int).diff().gt(0))[0]
    active_block_ends = np.where(data["in_active_bias_correction_block"].astype(int).diff().lt(0))[0]

    # Handle edge cases: If block starts but has no corresponding end
    if len(active_block_ends) < len(active_block_starts):
        active_block_ends = np.append(active_block_ends, len(data) - 1)  # Assume last index is end
    # Ensure equal start-end pairing
    active_block_starts = active_block_starts[:len(active_block_ends)]

    # Initialize right/left block lists
    right_block_starts, right_block_ends = [], []
    left_block_starts, left_block_ends = [], []

    for idx in range(len(active_block_starts)):
        start, end = active_block_starts[idx], active_block_ends[idx]

        # Ensure the end index is valid
        if start >= len(data) or end >= len(data):
            continue  # Skip invalid indices

        # Compute mean signed coherence in the block but only for trials where outcome is correct
        mean_signed_coh = np.nanmean(data["signed_coherence"].iloc[start:end+1][data["outcome"].iloc[start:end+1] == 1])

        # Classify the block based on sign
        if mean_signed_coh > 0:
            right_block_starts.append(start)
            right_block_ends.append(end)
        else:
            left_block_starts.append(start)
            left_block_ends.append(end)

    return data.index, right_block_starts, right_block_ends, left_block_starts, left_block_ends

def get_active_block_vars(data):
    if "in_active_bias_correction_block" not in data.columns:
        return data.index, None, None, None, None

    active_block_starts = np.where(data["in_active_bias_correction_block"].astype(int).diff().gt(0))[0]
    active_block_ends = np.where(data["in_active_bias_correction_block"].astype(int).diff().lt(0))[0]

    # Handle edge cases: If block starts but has no corresponding end
    if len(active_block_ends) < len(active_block_starts):
        active_block_ends = np.append(active_block_ends, len(data) - 1)  # Assume last index is end
    # Ensure equal start-end pairing
    active_block_starts = active_block_starts[:len(active_block_ends)]

    # Initialize right/left block lists
    right_block_starts, right_block_ends = [], []
    left_block_starts, left_block_ends = [], []

    for idx in range(len(active_block_starts)):
        start, end = active_block_starts[idx], active_block_ends[idx]

        # Ensure the end index is valid
        if start >= len(data) or end >= len(data):
            continue  # Skip invalid indices

        # Compute mean signed coherence in the block but only for trials where outcome is correct
        mean_signed_coh = np.nanmean(data["signed_coherence"].iloc[start:end+1][data["outcome"].iloc[start:end+1] == 1])

        # Classify the block based on sign
        if mean_signed_coh > 0:
            right_block_starts.append(start)
            right_block_ends.append(end)
        else:
            left_block_starts.append(start)
            left_block_ends.append(end)

    return data.index, right_block_starts, right_block_ends, left_block_starts, left_block_ends

def get_all_rolling_bias(all_data, window=20):
    rolling_bias = np.zeros(window)
    rolling_bias_idx = 0
    accumulated_rolling_bias = []

    is_active_correction_present = "in_active_bias_correction_block" in all_data.columns

    for trial in all_data.itertuples():
        if is_active_correction_present and trial.in_active_bias_correction_block:
            # Reset rolling bias during active correction
            rolling_bias.fill(0)
            rolling_bias_idx = 0
        elif not trial.is_correction_trial and trial.outcome is not None:
            # Update rolling bias for valid trials
            rolling_bias[rolling_bias_idx] = trial.choice
            rolling_bias_idx = (rolling_bias_idx + 1) % window

        # Store current rolling mean
        accumulated_rolling_bias.append(np.mean(rolling_bias))
    return accumulated_rolling_bias

def logistic(x, bias, sensitivity):
    return expit(sensitivity * (x - bias))

def fit_psychometric(x, y):
    # 📌 Fit a psychometric function for choices
    def loss(params):
        return -np.sum(y * np.log(logistic(x, *params)) + (1 - y) * np.log(1 - logistic(x, *params)))

    result = opt.minimize(loss, [0, 0.1])  # Initial guess: bias=0, sensitivity=0.1
    return result.x  # Returns bias and sensitivity

def get_sensory_noise(data):
    coherence,  choices = pmf_utils.get_psychometric_data(data, fit=False)
    bias_mouse, sensitivity_mouse = fit_psychometric(coherence, choices)
    sensory_noise_mouse = 1 / sensitivity_mouse  # Estimate noise level
    return sensory_noise_mouse

if __name__ == "__main__":
    # Get session information for recent sessions
    session_info = get_recent_sessions(last_X_business_days=30)
    session_info.date = pd.to_datetime(session_info.date).dt.date
    analyzed_data = {}

    # try to load old data if exists
    try:
        old_session_info = pd.read_csv(Path(PROCESSED_DATA_DIR / "session_info.csv"))
        # if old_session_info is not empty and same as session_info then stop processing
        if not old_session_info.empty:
            newly_added_sessions = set(old_session_info.session_uuid) - set(session_info.session_uuid)
            if not newly_added_sessions:
                print("No new data to process.")
                exit()
    except FileNotFoundError:
        pass

    # Process each mouse and session
    for mouse_id in session_info.mouse_id.unique():
        mouse_sessions = session_info[session_info.mouse_id == mouse_id]
        for idx_date, date in enumerate(mouse_sessions.date.unique()):
            sessions = mouse_sessions[mouse_sessions.date == date].reset_index()
            for idx, metadata in sessions.iterrows():
                trial_info = pd.read_csv(
                    SHARED_DATA_DIR / mouse_id / "data/random_dot_motion" / metadata.experiment / metadata.session / f"{mouse_id}_trial.csv"
                )
                trial_info, all_trial_info = preprocess_data(trial_info)

                condition = (session_info.mouse_id == mouse_id) & (session_info.date == date) & (session_info.session == metadata.session)

                # Skip sessions with less than 100 attempted trials
                if all_trial_info.shape[0] < 50:
                    # drop the row from session_info
                    session_info = session_info.drop(session_info[condition].index)
                    continue

                session_info.loc[condition, ["total_attempts", "total_valid", "session_accuracy", "total_reward", "sensory_noise"]] = [
                    max(trial_info.idx_attempt),
                    max(trial_info.idx_valid),
                    np.nanmean(trial_info.outcome) * 100,
                    np.sum(trial_info.trial_reward).astype(int),
                    get_sensory_noise(all_trial_info),
                ]

                coherences, accuracies = pmf_utils.get_accuracy_data(trial_info)
                _, reaction_time_median, reaction_time_mean, reaction_time_sd = pmf_utils.get_chronometric_data(trial_info)
                all_trial_idx, right_active_block_starts, right_active_block_ends, left_active_block_starts, left_active_block_ends = get_active_block_vars(all_trial_info)
                all_data_rolling_bias = get_all_rolling_bias(all_trial_info, window=20)

                analyzed_data[metadata.session_uuid] = {
                    "binned_trials": trial_info.idx_valid,
                    "binned_accuracies": np.array((100 * trial_info.outcome.rolling(window=20).mean())),
                    "rolling_trials": trial_info.idx_valid,
                    "rolling_bias": np.array(trial_info.choice.rolling(window=20).mean()),
                    "coherences": coherences,
                    "accuracy": accuracies,
                    "reaction_time_mean": reaction_time_mean,
                    "reaction_time_median": reaction_time_median,
                    "reaction_time_sd": reaction_time_sd,
                    # all trials rolling bias plot
                    "all_data_idx": all_trial_idx,
                    "all_data_rolling_bias": all_data_rolling_bias,
                    "all_data_choice": all_trial_info.choice,
                    "right_active_block_starts": right_active_block_starts,
                    "right_active_block_ends": right_active_block_ends,
                    "left_active_block_starts": left_active_block_starts,
                    "left_active_block_ends": left_active_block_ends,
                }

    # Save the updated DataFrame to a new CSV file
    session_info.to_csv(Path(PROCESSED_DATA_DIR / "session_info.csv"), index=False)
    # save master_dict as pickle file
    with open(Path(PROCESSED_DATA_DIR / "analyzed_data.pkl"), "wb") as f:
        pickle.dump(analyzed_data, f)
