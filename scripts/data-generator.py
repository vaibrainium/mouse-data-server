import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import dotenv
import os

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
    required_columns = ["date", "start_weight", "end_weight", "baseline_weight", "experiment", "session"]

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

        session_data.append(history[["mouse_id", "date", "start_weight", "end_weight", "experiment", "session"]])

    # Concatenate all results into a single DataFrame
    return pd.concat(session_data, ignore_index=True) if session_data else pd.DataFrame()


def preprocess_data(data):
    data = data.dropna(subset=["outcome"])
    data = data[data.is_correction_trial == False]
    if "in_active_bias_correction_block" in data.columns:
        data = data[data.in_active_bias_correction_block == False]
    return data

def get_binned_accuracy(data, bin_size=20):
    outcome_array = data["outcome"].astype(float).to_numpy()
    num_bins = len(outcome_array) // bin_size
    binned_accuracy = [np.nanmean(outcome_array[i * bin_size : (i + 1) * bin_size]) * 100 for i in range(num_bins)]
    binned_indices = np.arange(num_bins) * bin_size
    return binned_indices, np.array(binned_accuracy)

def get_binned_accuracy(data, bin_size=20):
    outcome_array = data["outcome"].astype(float).to_numpy()
    num_bins = len(outcome_array) // bin_size
    binned_accuracy = [np.nanmean(outcome_array[i * bin_size : (i + 1) * bin_size]) * 100 for i in range(num_bins)]
    binned_indices = np.arange(num_bins) * bin_size
    return binned_indices, np.array(binned_accuracy)

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
            # sort both dataframes by date and mouse_id
            old_session_info = old_session_info.sort_values(by=["date", "mouse_id"]).reset_index(drop=True)
            session_info = session_info.sort_values(by=["date", "mouse_id"]).reset_index(drop=True)
            # check if 'mouse_id', 'date', 'session', and 'experiment' are the same in both dataframes
            columns_to_compare = ["mouse_id", "experiment", "session"]
            if session_info[columns_to_compare].equals(old_session_info[columns_to_compare]):
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
                trial_info = preprocess_data(trial_info)

                condition = (session_info.mouse_id == mouse_id) & (session_info.date == date) & (session_info.session == metadata.session)

                session_info.loc[condition, ["total_attempts", "total_valid", "session_accuracy", "total_reward"]] = [
                    max(trial_info.idx_attempt),
                    max(trial_info.idx_valid),
                    np.nanmean(trial_info.outcome) * 100,
                    np.sum(trial_info.trial_reward).astype(int),
                ]

                binned_trial, binned_accuracy = get_binned_accuracy(trial_info, bin_size=20)
                coherences, accuracies = pmf_utils.get_accuracy_data(trial_info)
                _, reaction_time_median, reaction_time_mean, reaction_time_sd = pmf_utils.get_chronometric_data(trial_info)

                analyzed_data[metadata["index"]] = {
                    "binned_trials": trial_info.idx_valid,
                    "binned_accuracies": np.array((100 * trial_info.outcome.rolling(window=20).mean())),
                    "rolling_trials": trial_info.idx_valid,
                    "rolling_bias": np.array(trial_info.choice.rolling(window=20).mean()),
                    "coherences": coherences,
                    "accuracy": accuracies,
                    "reaction_time_mean": reaction_time_mean,
                    "reaction_time_median": reaction_time_median,
                    "reaction_time_sd": reaction_time_sd,
                }

    # Save the updated DataFrame to a new CSV file
    session_info.to_csv(Path(PROCESSED_DATA_DIR / "session_info.csv"), index=False)
    # save master_dict as pickle file
    with open(Path(PROCESSED_DATA_DIR / "analyzed_data.pkl"), "wb") as f:
        pickle.dump(analyzed_data, f)
