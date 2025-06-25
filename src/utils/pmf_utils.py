import warnings

import numpy as np
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin

# Suppress warnings
warnings.filterwarnings("ignore")


class PsychometricFunction(BaseEstimator, RegressorMixin):
	"""Fits a logistic regression model for psychometric analysis.

	Models:
		- "logit_2": Logistic function with mean and variance (without lapse rate).
		- "logit_3": Logistic function with lapse rate.
		- "logit_4": Logistic function with lapse and guess rates.
	"""

	def __init__(
		self,
		model="logit_3",
		mean_lims=(-100, 100),
		var_lims=(1e-5, 30),
		lapse_rate_lims=(0, 0.1),
		guess_rate_lims=(0, 0.1),
	) -> None:
		self.model = model
		self.mean_lims = mean_lims
		self.var_lims = var_lims
		self.lapse_rate_lims = lapse_rate_lims
		self.guess_rate_lims = guess_rate_lims

		if model not in ["logit_2", "logit_3", "logit_4"]:
			msg = f"Unknown model: {model}. Choose 'logit_2', 'logit_3' or 'logit_4'."
			raise ValueError(msg)

	def logit_2(self, x, mean, var):
		"""Logistic function with only mean and variance."""
		return 1 / (1 + np.exp(-var * (x - mean)))

	def logit_3(self, x, mean, var, lapse_rate):
		"""Logistic function with lapse rate."""
		return lapse_rate + ((1 - 2 * lapse_rate) / (1 + np.exp(-var * (x - mean))))

	def logit_4(self, x, mean, var, lapse_rate, guess_rate):
		"""Logistic function with lapse and guess rates."""
		return lapse_rate + ((1 - guess_rate - lapse_rate) / (1 + np.exp(-var * (x - mean))))
	def fit(self, x, y, trial_counts=None):
		"""Fit the psychometric function to data, incorporating trial counts as weights."""
		# Remove NaNs
		mask = ~np.isnan(y)
		x, y = x[mask], y[mask]
		if trial_counts is not None:
			trial_counts = trial_counts[mask]

		# Store original x data for rescaling parameters back
		self.x_min_ = np.min(x)
		self.x_max_ = np.max(x)
		self.x_range_ = self.x_max_ - self.x_min_

		# Normalize x to [0, 1] for numerical stability
		if self.x_range_ > 0:
			x_norm = (x - self.x_min_) / self.x_range_
		else:
			x_norm = x  # Handle case where all x values are the same

		# Choose fitting function based on model
		if self.model == "logit_2":
			self._fit_func = self.logit_2
			param_lims = [self.mean_lims, self.var_lims]  # Only mean and variance
		elif self.model == "logit_3":
			self._fit_func = self.logit_3
			param_lims = [self.mean_lims, self.var_lims, self.lapse_rate_lims]
		elif self.model == "logit_4":
			self._fit_func = self.logit_4
			param_lims = [self.mean_lims, self.var_lims, self.lapse_rate_lims, self.guess_rate_lims]
		# Normalize parameter bounds for fitting
		mean_lims_norm = ((self.mean_lims[0] - self.x_min_) / self.x_range_,
		                 (self.mean_lims[1] - self.x_min_) / self.x_range_) if self.x_range_ > 0 else (0, 1)
		var_lims_norm = (self.var_lims[0] * self.x_range_, self.var_lims[1] * self.x_range_) if self.x_range_ > 0 else self.var_lims

		# Set normalized parameter limits based on model
		if self.model == "logit_2":
			param_lims_norm = [mean_lims_norm, var_lims_norm]
		elif self.model == "logit_3":
			param_lims_norm = [mean_lims_norm, var_lims_norm, self.lapse_rate_lims]  # Lapse rate doesn't need scaling
		else:  # logit_4
			param_lims_norm = [mean_lims_norm, var_lims_norm, self.lapse_rate_lims, self.guess_rate_lims]

		bounds = list(zip(*param_lims_norm, strict=False))
		# Better initial guess using data
		initial_mean = 0.5  # Middle of normalized range
		initial_var = 4.0   # Reasonable slope for normalized data
		initial_guess = [initial_mean, initial_var]

		if self.model in ("logit_3", "logit_4"):
			initial_guess.append(np.mean(self.lapse_rate_lims))  # Better than min
		if self.model == "logit_4":
			initial_guess.append(np.mean(self.guess_rate_lims))  # Better than min

		# Compute weights: More trials → Higher weight
		if trial_counts is not None:
			weights = np.sqrt(trial_counts)  # Square root scaling to balance influence
			weights = np.maximum(weights, 1e-10)  # Avoid division by zero
			sigma = 1 / weights  # Use inverse for curve fitting
		else:
			sigma = None  # No weighting if trial_counts is not provided

		# Fit using weighted least squares (WLS) on normalized data
		popt, pcov = curve_fit(self._fit_func, x_norm, y, p0=initial_guess, bounds=bounds, sigma=sigma, absolute_sigma=False, maxfev=10000)

		# Rescale parameters back to original scale
		mean_orig = popt[0] * self.x_range_ + self.x_min_ if self.x_range_ > 0 else popt[0]
		var_orig = popt[1] / self.x_range_ if self.x_range_ > 0 else popt[1]

		# Store results in original scale
		self.coefs_ = {"mean": mean_orig, "var": var_orig}
		if self.model in ("logit_3", "logit_4"):
			self.coefs_["lapse_rate"] = popt[2]  # No rescaling needed
		if self.model == "logit_4":
			self.coefs_["guess_rate"] = popt[3]  # No rescaling needed

		# Rescale covariance matrix (approximate)
		scale_factors = np.array([self.x_range_ if i == 0 else (1/self.x_range_ if i == 1 else 1)
		                         for i in range(len(popt))])
		if self.x_range_ > 0:
			self.covar_ = pcov * np.outer(scale_factors, scale_factors)
		else:
			self.covar_ = pcov

		return self
	def predict(self, x):
		"""Predict using the fitted model."""
		# The coefficients are already in original scale, so we can use them directly
		# with original scale x data
		return self._fit_func(x, **self.coefs_)


def fit_psychometric_function(x_data, y_data, trial_counts, model_type, **kwargs):
	"""Fit psychometric function with user-defined or default parameters."""
	params = {
		"model": model_type,
		"mean_lims": (-100, 100),
		"var_lims": (1e-5, 30),
		"lapse_rate_lims": (1e-5, 0.2),
		"guess_rate_lims": (1e-5, 0.2),
		**kwargs,  # Override defaults with user input
	}
	return PsychometricFunction(**params).fit(x_data, y_data, trial_counts)


def get_psychometric_data(data, positive_direction="right", fit=True, model_type="logit_3", **kwargs):
	"""Extracts psychometric data and optionally fits a model."""
	unique_coh = np.unique(data["signed_coherence"])
	x_data = np.where(positive_direction == "left", -unique_coh, unique_coh)
	y_data = []
	trial_counts = []
	for coh in unique_coh:
		mask = data["signed_coherence"] == coh
		total_trials = np.sum(mask)

		if total_trials == 0:
			continue  # Skip coherence levels with no trials

		prop_positive = np.mean(data["choice"][mask] == (positive_direction == "right"))
		y_data.append(prop_positive)
		trial_counts.append(total_trials)

	# Convert to numpy arrays and sort
	x_data, y_data, trial_counts = map(np.array, zip(*sorted(zip(x_data, y_data, trial_counts, strict=False)), strict=False))

	if not fit:
		return x_data, y_data
	# Fit the model using the specified model_type (logit_2, logit_3, logit_4)
	model = fit_psychometric_function(x_data, y_data, trial_counts=trial_counts, model_type=model_type, **kwargs)
	x_model = np.linspace(min(x_data), max(x_data), 100)
	y_model = model.predict(x_model)

	return x_data, y_data, model, x_model, y_model


def get_chronometric_data(data, positive_direction="right"):
	"""Computes reaction time statistics for different coherence levels."""
	unique_coh = np.unique(data["signed_coherence"])
	coherences, rt_median, rt_mean, rt_sd = [], [], [], []

	for coh in unique_coh:
		trials = data[(data["signed_coherence"] == coh) & (data["outcome"] == 1)]
		if trials.empty:
			continue

		coherences.append(coh if positive_direction == "right" else -coh)
		rt_median.append(np.median(trials["response_time"]))
		rt_mean.append(np.mean(trials["response_time"]))
		rt_sd.append(np.std(trials["response_time"]))

	return map(np.array, zip(*sorted(zip(coherences, rt_median, rt_mean, rt_sd, strict=False)), strict=False))


def get_accuracy_data(data, positive_direction="right"):
	"""Computes accuracy across coherence levels."""
	unique_coh = np.unique(data["signed_coherence"])
	coherences = np.where(positive_direction == "left", -unique_coh, unique_coh)
	accuracy = np.array([np.mean(data["outcome"][data["signed_coherence"] == coh] == 1) for coh in unique_coh])

	return map(np.array, zip(*sorted(zip(coherences, accuracy, strict=False)), strict=False))
