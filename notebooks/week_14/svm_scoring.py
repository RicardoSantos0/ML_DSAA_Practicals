"""Week 14 scoring policy for models whose inverse target transform can overflow."""

import warnings

import numpy as np
from sklearn.metrics import mean_absolute_error


def negative_mae_if_finite(estimator, X, y):
    """Return negative MAE, or negative infinity for any nonfinite prediction."""
    with np.errstate(over="ignore", invalid="ignore"):
        prediction = estimator.predict(X)
    if not np.all(np.isfinite(prediction)):
        return -np.inf
    return -mean_absolute_error(y, prediction)


def ignore_unknown_category_warning():
    """Hide only the encoder notice for its configured unknown-level behaviour."""
    warnings.filterwarnings(
        "ignore",
        message=r"^Found unknown categories in columns \[.*\] during transform\. These unknown categories will be encoded as all zeros$",
        category=UserWarning,
        module=r"^sklearn\.preprocessing\._encoders$",
    )


def ignore_svc_search_warnings():
    """Apply the declared bounded-SVC warning policy in search workers only."""
    from sklearn.exceptions import ConvergenceWarning
    ignore_unknown_category_warning()
    warnings.filterwarnings("ignore", message=r"^Solver terminated early \(max_iter=.*",
        category=ConvergenceWarning, module=r"^sklearn\.svm\._base$")


from contextlib import contextmanager


@contextmanager
def expected_nonfinite_search_warning():
    """Hide the search notice only for the deliberate negative-infinity policy."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(?s)^One or more of the test scores are non-finite: \[(?=[^\]]*-inf)(?![^\]]*\bnan\b)(?![^\]]*(?<!-)inf)[^\]]*\]$",
            category=UserWarning,
            module=r"^sklearn\.model_selection\._search$",
        )
        yield
