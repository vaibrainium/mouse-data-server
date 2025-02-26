import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from sklearn.base import BaseEstimator, RegressorMixin

# suppress warnings
warnings.filterwarnings("ignore")


class PsychometricFunction(BaseEstimator, RegressorMixin):
    """
    Fit a logistic regression (Logit or it's modifications)

    Model parameters:
        - mean (bias): The mean of logistic distribution.
        - variance (threshold): Spread/slope of distribution.
        - lapse_rate: Free parameter to indicate deviation in accuracy from 100% for easiest stimulus condition (at 0 and 1 of x-axis). Can either be same for both choices or different for each choice.
        - guess_rate: Same as lapse rate for lower bound.

    For all parameters, starting point is set as a mean of the upper and lower limit.
    :param mean_lims: Default (-100, 100).
    :param var_lims: Default (0.001, 20).
    :param lapse_rate_lims: Default (0.01, 0.2).
    :param guess_rate_lims: Default (0.01, 0.2).
    """

    # def __init__(self, model= "logit_3", mean_lims = (-100, 100), var_lims = (.001, 20), lapse_rate_lims = (.01,.2), guess_rate_lims = (.01,.2)) -> None:
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

    def __post_init__(self) -> None:
        if self.model not in ["logit_3", "logit_4"]:
            raise ValueError(f"Unknown mode: {self.model}. Available models: 'logit_4', 'logit_4'")

    def logit_3(self, x: np.ndarray, mean: float, var: float, lapse_rate: float) -> np.ndarray:
        return lapse_rate + ((1.0 - 2 * lapse_rate) / (1 + np.exp(-var * (x - mean))))

    def logit_4(
        self,
        x: np.ndarray,
        mean: float,
        var: float,
        lapse_rate: float,
        guess_rate: float,
    ) -> np.ndarray:
        return lapse_rate + ((1.0 - guess_rate - lapse_rate) / (1 + np.exp(-var * (x - mean))))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PsychometricFunction":

        # Find nan values in y and remove them from x and y at the same index
        nan_indices = np.argwhere(np.isnan(y))
        x = np.delete(x, nan_indices)
        y = np.delete(y, nan_indices)

        if self.model.lower() == "logit_3":
            self._fit_func = self.logit_3
            lims = [self.mean_lims, self.var_lims, self.lapse_rate_lims]
            bounds = (
                [
                    self.mean_lims[0],
                    self.var_lims[0],
                    self.lapse_rate_lims[0],
                ],
                [
                    self.mean_lims[1],
                    self.var_lims[1],
                    self.lapse_rate_lims[1],
                ],
            )

        else:
            self._fit_func = self.logit_4
            lims = [
                self.mean_lims,
                self.var_lims,
                self.lapse_rate_lims,
                self.guess_rate_lims,
            ]
            bounds = (
                [
                    self.mean_lims[0],
                    self.var_lims[0],
                    self.lapse_rate_lims[0],
                    self.guess_rate_lims[0],
                ],
                [
                    self.mean_lims[1],
                    self.var_lims[1],
                    self.lapse_rate_lims[1],
                    self.guess_rate_lims[1],
                ],
            )

        popt, pcov = optimize.curve_fit(
            f=self._fit_func,
            xdata=x,
            ydata=y,
            p0=[np.min(lim) for lim in lims],
            bounds=bounds,
        )

        self.coefs_ = {"mean": popt[0], "var": popt[1], "lapse_rate": popt[2]}
        if self.model.lower() == "logit_4":
            self.coefs_.update({"guess_rate": popt[3]})

        self.covar_ = pcov

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._fit_func(x, **self.coefs_)

    def plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        show: bool = False,
        y_label: str = "Prop. of Positive choices",
        x_label: str = "Coherence",
    ):
        fig, ax = plt.subplots()

        if y is not None:
            ax.scatter(x, y, label="y")

        x = np.arange(-100, 100, 0.1)
        ax.plot(x, self.predict(x), label="y_pred")

        ax.set_xlim(-100, 100)
        ax.set_ylim(0, 1)

        ax.legend()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.tight_layout()

        if show:
            fig.show()

        return fig


def get_psychometric_data(data, positive_direction="right"):
    x_data = np.asarray([])
    y_data = np.asarray([])
    for _, coh in enumerate(np.unique(data["signed_coherence"])):
        if positive_direction == "right":
            x_data = np.append(x_data, coh)
            y_data = np.append(y_data, np.sum(data["choice"][data["signed_coherence"] == coh] == 1) / np.sum(data["signed_coherence"] == coh))
        elif positive_direction == "left":
            x_data = np.append(x_data, -coh)
            y_data = np.append(y_data, np.sum(data["choice"][data["signed_coherence"] == coh] == 0) / np.sum(data["signed_coherence"] == coh))
    # sorting
    x_data, y_data = zip(*sorted(zip(x_data, y_data)))

    # fit psychometric function
    x_model = np.linspace(min(x_data), max(x_data), 100)
    model = fit_psychometric_function(x_data, y_data)
    y_model = model.predict(x_model)
    return np.asarray(x_data), np.asarray(y_data), model, np.asarray(x_model), np.asarray(y_model)


def fit_psychometric_function(x_data, y_data, **model_kwargs):
    defaults = {"model": "logit_4", "var_lims": (1e-5, 10), "lapse_rate_lims": (1e-5, 0.2), "guess_rate_lims": (1e-5, 0.2)}
    for k, v in defaults.items():
        val = model_kwargs.get(k, v)
        model_kwargs[k] = val
    model = PsychometricFunction(**model_kwargs).fit(x_data, y_data)
    return model


def get_psychometric_data(data, positive_direction="right", fit=True):
    x_data, y_data = [], []

    for coh in np.unique(data["signed_coherence"]):
        mask = data["signed_coherence"] == coh
        total_trials = np.sum(mask)

        if total_trials == 0:  # Avoid division by zero
            continue

        if positive_direction == "right":
            proportion_positive = np.sum(data["choice"][mask] == 1) / total_trials
            x_data.append(coh)
        elif positive_direction == "left":
            proportion_positive = np.sum(data["choice"][mask] == 0) / total_trials
            x_data.append(-coh)  # Flip sign for leftward motion

        y_data.append(proportion_positive)

    # Sort data
    x_data, y_data = zip(*sorted(zip(x_data, y_data)))
    x_data, y_data = np.array(x_data), np.array(y_data)

    if not fit:
        return x_data, y_data

    # fit psychometric function
    x_model = np.linspace(min(x_data), max(x_data), 100)
    model = fit_psychometric_function(x_data, y_data)
    y_model = model.predict(x_model)
    return  x_data, y_data, model, np.asarray(x_model), np.asarray(y_model)


def fit_psychometric_function(x_data, y_data, **model_kwargs):
    defaults = {"model": "logit_4", "var_lims": (1e-5, 10), "lapse_rate_lims": (1e-5, 0.2), "guess_rate_lims": (1e-5, 0.2)}
    for k, v in defaults.items():
        val = model_kwargs.get(k, v)
        model_kwargs[k] = val
    model = PsychometricFunction(**model_kwargs).fit(x_data, y_data)
    return model


def get_chronometric_data(data, positive_direction="right"):
    coherences = []
    reaction_time_median = []
    reaction_time_mean = []
    reaction_time_sd = []

    for coh in np.unique(data["signed_coherence"]):
        subset = data[(data["signed_coherence"] == coh) & (data["outcome"] == 1)]  # Correct trials

        if subset["response_time"].empty:
            continue  # Skip coherence levels with no correct trials

        if positive_direction == "right":
            coherences.append(coh)
        elif positive_direction == "left":
            coherences.append(-coh)  # Flip sign for leftward motion

        # Compute reaction time statistics
        reaction_time_median.append(np.median(subset["response_time"]))
        reaction_time_mean.append(np.mean(subset["response_time"]))
        reaction_time_sd.append(np.std(subset["response_time"]))

    # Convert to numpy arrays and sort
    sorted_data = sorted(zip(coherences, reaction_time_median, reaction_time_mean, reaction_time_sd))
    coherences, reaction_time_median, reaction_time_mean, reaction_time_sd = map(np.array, zip(*sorted_data))

    return coherences, reaction_time_median, reaction_time_mean, reaction_time_sd


def get_accuracy_data(data, positive_direction="right"):
    coherences = np.asarray([])
    accuracy = np.asarray([])
    for _, coh in enumerate(np.unique(data["signed_coherence"])):
        if positive_direction == "right":
            coherences = np.append(coherences, coh)
            accuracy = np.append(accuracy, np.sum(data["outcome"][data["signed_coherence"] == coh] == 1) / np.sum(data["signed_coherence"] == coh))
        elif positive_direction == "left":
            coherences = np.append(coherences, -coh)
            accuracy = np.append(accuracy, np.sum(data["outcome"][data["signed_coherence"] == coh] == 1) / np.sum(data["signed_coherence"] == coh))

    sorted_data = sorted(zip(coherences, accuracy))
    coherences, accuracy = map(np.array, zip(*sorted_data))
    return coherences, accuracy
