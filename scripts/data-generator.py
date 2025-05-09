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
if not SHARED_DATA_DIR or not PROCESSED_DATA_DIR:
	raise ValueError("Environment variables SHARED_DATA_DIR and PROCESSED_DATA_DIR must be set.")

def get_recent_sessions(last_X_business_days=None, start_date=None, end_date=None) -> pd.DataFrame:
	"""
	Get session information for recent sessions based on the specified date range or last X business days.

	Parameters
	----------
	last_X_business_days : int, optional
	start_date : str, optional
	end_date : str, optional

	Returns
	-------
	pd.DataFrame
	"""
	# Get all mouse ID folders except "XXX"
	mouse_ids: list[Path] = [f for f in SHARED_DATA_DIR.iterdir() if f.is_dir() and f.name != "XXX"]
	today: pd.Timestamp = pd.to_datetime("today")  # Keep full timestamp

	if last_X_business_days:
		start_date, end_date = today - pd.offsets.BDay(last_X_business_days), today
	else:
		start_date, end_date = map(pd.to_datetime, (start_date, end_date or today))
		if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
			raise ValueError("Invalid start_date or end_date format.")

	session_data = []
	required_cols = ["date", "time", "start_weight", "end_weight", "baseline_weight", "protocol", "experiment", "session", "session_uuid"]

	for mouse in mouse_ids:
		history_path = mouse / "history.csv"
		if not history_path.exists():
			continue

		history = pd.read_csv(history_path, parse_dates=["date"], dayfirst=True).dropna(subset=["date"])
		history["date"] = pd.to_datetime(history["date"], format='mixed', errors="coerce")
		history = history[(history["date"] >= start_date) & (history["date"] <= end_date)]
		if history.empty:
			continue

		history["time"] = pd.to_datetime(history["date"], format='%H:%M:%S', errors="coerce").dt.time
		history = history.reindex(columns=required_cols, fill_value=pd.NA)
		history[["start_weight", "end_weight"]] = history[["start_weight", "end_weight"]].div(history["baseline_weight"], axis=0) * 100
		history.insert(0, "mouse_id", mouse.name)
		session_data.append(history)

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

	active_blocks = data["in_active_bias_correction_block"].astype(int).diff()
	starts, ends = np.where(active_blocks > 0)[0], np.where(active_blocks < 0)[0]
	if len(ends) < len(starts):
		ends = np.append(ends, len(data) - 1)

	left_starts, left_ends, right_starts, right_ends = [], [], [], []
	for s, e in zip(starts, ends):
		if s >= len(data) or e >= len(data):
			continue
		mean_signed_coh = np.nanmean(data.loc[s:e, "signed_coherence"][data.loc[s:e, "outcome"] == 1])
		(right_starts if mean_signed_coh > 0 else left_starts).append(s)
		(right_ends if mean_signed_coh > 0 else left_ends).append(e)

	return data.index, right_starts, right_ends, left_starts, left_ends

def get_all_rolling_bias(data, window=20):
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
	bias, sensitivity = fit_psychometric(coherence, choices)
	sensory_noise = 1 / sensitivity  # Estimate noise level
	return sensory_noise

def load_existing_data():
	"""Load existing session and analyzed data if available."""
	old_session_file = Path(PROCESSED_DATA_DIR / "session_info.csv")
	old_analyzed_file = Path(PROCESSED_DATA_DIR / "analyzed_data.pkl")

	# Check if both session info and analyzed data exist
	if old_session_file.exists() and old_analyzed_file.exists():
		old_session_info = pd.read_csv(old_session_file)
		with open(old_analyzed_file, "rb") as f:
			old_analyzed_data = pickle.load(f)
	else:
		old_session_info = pd.DataFrame()  # No previous data
		old_analyzed_data = {}

	return old_session_info, old_analyzed_data

def update_sessions(old_session_info, old_analyzed_data, session_info):
	"""Remove outdated sessions from the old session info."""
	sessions_to_remove = old_session_info[~old_session_info["session_uuid"].isin(session_info["session_uuid"])]["session_uuid"].tolist()
	sessions_to_add = session_info[~session_info["session_uuid"].isin(old_session_info["session_uuid"])]

	if sessions_to_remove:
		updated_session_info = session_info[~session_info["session_uuid"].isin(sessions_to_remove)].reset_index(drop=True)
		for key in sessions_to_remove:
			old_analyzed_data.pop(key, None)
		updated_analyzed_data = old_analyzed_data
	else:
		updated_session_info = session_info
		updated_analyzed_data = old_analyzed_data
	return updated_session_info, updated_analyzed_data, sessions_to_add

def process_analyzed_data(all_trial_info, valid_trial_info):
	coherences, accuracies = pmf_utils.get_accuracy_data(valid_trial_info)
	_, reaction_time_median, reaction_time_mean, reaction_time_sd = pmf_utils.get_chronometric_data(valid_trial_info)
	all_trial_idx, right_active_block_starts, right_active_block_ends, left_active_block_starts, left_active_block_ends = get_active_block_vars(all_trial_info)
	all_data_rolling_bias = get_all_rolling_bias(all_trial_info, window=20)

	return {
		"binned_trials": valid_trial_info.idx_valid,
		"binned_accuracies": np.array((100 * valid_trial_info.outcome.rolling(window=20).mean())),
		"rolling_trials": valid_trial_info.idx_valid,
		"rolling_bias": np.array(valid_trial_info.choice.rolling(window=20).mean()),
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

if __name__ == "__main__":
	# ✅ Load existing session data if available
	old_session_info, old_analyzed_data = load_existing_data()

	session_info = get_recent_sessions(last_X_business_days=30)
	session_info.date = pd.to_datetime(session_info.date).dt.date
	if session_info.empty:
		print("No session data found. Exiting.")
		exit()
	new_sessions = session_info
	analyzed_data = {}
	# ✅ Update sessions
	if old_session_info.empty:
		new_sessions = session_info
		analyzed_data = {}
	else:
		session_info, analyzed_data, new_sessions = update_sessions(old_session_info, old_analyzed_data, session_info)
		if new_sessions.empty:
			print("No new sessions to process.")
			exit()

	# Process each mouse and session
	for mouse_id in new_sessions.mouse_id.unique():
		mouse_sessions = new_sessions[new_sessions.mouse_id == mouse_id]
		for idx_date, date in enumerate(mouse_sessions.date.unique()):
			sessions = mouse_sessions[mouse_sessions.date == date].reset_index()
			for idx, metadata in sessions.iterrows():
				trial_info = pd.read_csv(
					SHARED_DATA_DIR / mouse_id / "data" / metadata.protocol / metadata.experiment / metadata.session / f"{mouse_id}_trial.csv"
				)
				valid_trial_info, all_trial_info = preprocess_data(trial_info)
				condition = (new_sessions.mouse_id == mouse_id) & (new_sessions.date == date) & (new_sessions.session == metadata.session)
				# Skip sessions with less than 50 attempted trials
				if all_trial_info.shape[0] < 50:
					new_sessions = new_sessions.loc[~condition]
					continue

				summary_info = pd.read_csv(
					SHARED_DATA_DIR / mouse_id / "data" / metadata.protocol / metadata.experiment / f"{mouse_id}_summary.csv"
				)
				summary_row = summary_info[summary_info.session_uuid == metadata.session_uuid]

				new_sessions.loc[condition, ["total_attempts", "total_valid", "session_accuracy", "left_accuracy", "right_accuracy", "total_reward", "sensory_noise", "comments"]] = [
					max(valid_trial_info.idx_attempt),
					max(valid_trial_info.idx_valid),
					np.nanmean(valid_trial_info.outcome) * 100,
					np.nanmean(valid_trial_info.outcome[np.sign(valid_trial_info.signed_coherence) == -1]) * 100,
					np.nanmean(valid_trial_info.outcome[np.sign(valid_trial_info.signed_coherence) == 1]) * 100,
					np.sum(valid_trial_info.trial_reward).astype(int),
					get_sensory_noise(all_trial_info),
					summary_row.comments.values[0] if not summary_row.empty else None,
				]

				coherences, accuracies = pmf_utils.get_accuracy_data(valid_trial_info)
				_, reaction_time_median, reaction_time_mean, reaction_time_sd = pmf_utils.get_chronometric_data(valid_trial_info)
				all_trial_idx, right_active_block_starts, right_active_block_ends, left_active_block_starts, left_active_block_ends = get_active_block_vars(all_trial_info)
				all_data_rolling_bias = get_all_rolling_bias(all_trial_info, window=20)

				analyzed_data[metadata.session_uuid] = process_analyzed_data(all_trial_info, valid_trial_info)


	updated_session_info = pd.concat([old_session_info, new_sessions], ignore_index=True)
	updated_session_info.drop_duplicates(subset=["session_uuid"], keep="last", inplace=True)
	updated_session_info.to_csv(PROCESSED_DATA_DIR / "session_info.csv", index=False)
	# ✅ Save updated analyzed data
	with open(PROCESSED_DATA_DIR / "analyzed_data.pkl", "wb") as f:
		pickle.dump(analyzed_data, f)

	print(f"Processing complete. {len(new_sessions)} new sessions added.")
