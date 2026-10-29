"""Apply a logged preprocessing recipe inside each training fold.

`load_recipe` returns a frame and the recipe its log configures, and
`FoldPreprocessor` fits that recipe on training rows and transforms any rows.
"""

from __future__ import annotations

import ast
import importlib
import json
import operator
from functools import partial
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from category_encoders import CountEncoder, TargetEncoder
from scipy.stats import chi2_contingency, pearsonr, pointbiserialr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted


# -------------------------------------------------------------------------
# Reading a recipe and its frame
# -------------------------------------------------------------------------

def _decision(carries, *required):
    """Return the one logged entry holding every required key."""
    matches = [entry for entry in carries
               if all(key in entry for key in required)]
    if len(matches) != 1:
        raise ValueError(f"expected one entry holding {required}, got {len(matches)}")
    return matches[0]

# The arithmetic a logged formula may use.
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def _rebuild(expression, names):
    """Build a logged call or formula from its parse tree, never executing it.

    `names` maps each bare name to a class or a column.
    """
    node = (ast.parse(expression, mode="eval").body
            if isinstance(expression, str) else expression)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.args:
            raise ValueError(f"a logged call is one name with keywords: {ast.unparse(node)}")
        return names(node.func.id)(
            **{keyword.arg: _rebuild(keyword.value, names) for keyword in node.keywords}
        )
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_rebuild(node.left, names),
                                        _rebuild(node.right, names))
    if isinstance(node, ast.Name):
        return names(node.id)
    return ast.literal_eval(node)

def load_recipe(data_path, log_path):
    """Return the frame at `data_path` and the recipe configured by `log_path`.

    Columns the log defines by a formula are added to the frame.
    """
    steps = json.loads(Path(log_path).read_text(encoding="utf-8"))["steps"]
    carries = [step["carries"] for step in steps if step["carries"] is not None]
    dtypes = [entry.get("cast", entry.get("dtypes")) for entry in carries
              if "cast" in entry or "dtypes" in entry]
    if len(dtypes) != 1:
        raise ValueError(f"expected one entry holding the dtypes, got {len(dtypes)}")
    frame = pd.read_csv(data_path, dtype=dtypes[0])
    for entry in carries:
        if "formula" in entry:
            frame[entry["feature"]] = _rebuild(entry["formula"], frame.__getitem__)

    fill = _decision(carries, "fill", "numeric", "categorical")
    transform = _decision(carries, "transform", "columns")
    representation = _decision(carries, "encoding", "scaler")
    if transform["transform"] not in (None, "none", "log1p"):
        raise ValueError(f"unknown transform: {transform['transform']!r}")
    recipe = {
        "numeric": list(representation["numeric"]),
        "categorical": list(representation["categorical"]),
        "dropped": list(representation.get("dropped", [])),
        "fill": dict(fill["fill"]),
        "log1p": (list(transform["columns"])
                  if transform["transform"] == "log1p" else []),
        "outliers": transform["outliers"],
        "encoding": representation["encoding"],
        "drop": representation["drop"],
        "scaler": representation["scaler"],
        "strategy": _decision(carries, "strategy")["strategy"],
    }
    return frame, recipe


# -------------------------------------------------------------------------
# Turning a recipe into an unfitted preprocessor
# -------------------------------------------------------------------------

# Encoders and scalers by the names a recipe uses; each fit clones them.
ENCODERS = {
    "one-hot": OneHotEncoder(handle_unknown="ignore", drop="first",
                             sparse_output=False),
    "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1),
    "count": CountEncoder(normalize=True, handle_unknown=0,
                          handle_missing="value"),
    "target": TargetEncoder(handle_unknown="value", handle_missing="value"),
}
SCALERS = {"none": None, "standard": StandardScaler,
           "min-max": MinMaxScaler, "robust": RobustScaler}

# The training quantiles each outlier treatment clips to; None clips nothing.
OUTLIER_QUANTILES = {"keep": None, "keep everything": None,
                     "winsorize 1/99": (0.01, 0.99)}

class ProportionalImputer:
    """Fill each gap with a label drawn in its column's training proportions."""

    def __init__(self, random_state=None):
        """Store the seed of the draws."""
        self.random_state = random_state

    def fit(self, frame):
        """Learn each column's label proportions."""
        self.shares = {c: frame[c].value_counts(normalize=True) for c in frame}
        return self

    def transform(self, frame):
        """Fill the gaps with labels drawn from the learned proportions."""
        draws = np.random.default_rng(self.random_state)
        frame = frame.copy()
        for column, shares in self.shares.items():
            gaps = frame[column].isna()
            if gaps.any():
                frame.loc[gaps, column] = draws.choice(
                    shares.index, size=gaps.sum(), p=shares.to_numpy())
        return frame

class FoldPreprocessor(TransformerMixin, BaseEstimator):
    """Fit a recipe's preprocessing on training rows and apply it to any rows."""

    def __init__(self, recipe, *, selection="inherit"):
        """Store the recipe; `selection=None` keeps every encoded column."""
        self.recipe = recipe
        self.selection = selection

    def _categorical_block(self, frame, fit=False):
        """Fill the label columns."""
        # scikit-learn imputers read np.nan as missing, but not pd.NA.
        block = frame[self.categorical_].astype(object).fillna(np.nan).infer_objects()
        if self.recipe["fill"]["categorical"] == "own level":
            # The new label and boolean columns need one common type.
            block = block.astype("string").astype(object).fillna(np.nan)
        if fit:
            self.categorical_imputer_.fit(block)
        return pd.DataFrame(self.categorical_imputer_.transform(block),
                            columns=self.categorical_, index=frame.index)

    def _numeric_block(self, frame, fit=False):
        """Clip, fill and log1p the numeric columns."""
        block = frame[self.numeric_].astype(object).fillna(np.nan).infer_objects()
        if fit:
            treatment = self.recipe["outliers"]
            if treatment.startswith("drop"):
                raise ValueError(
                    f"{treatment!r} removes training rows, which a transformer"
                    " cannot do; drop them in the fold loop, where y is available"
                )
            # Quantiles come from the observed values, before filling.
            limits = OUTLIER_QUANTILES[treatment]
            self.outlier_bounds_ = (None if limits is None
                                    else (block.quantile(limits[0]),
                                          block.quantile(limits[1])))
        if self.outlier_bounds_ is not None:
            block = block.clip(*self.outlier_bounds_, axis=1)
        if self.standardiser_ is None:
            if fit:
                self.numeric_imputer_.fit(block)
            filled = self.numeric_imputer_.transform(block)
        else:
            # KNN imputes on standardised columns, then returns to the units.
            if fit:
                self.numeric_imputer_.fit(self.standardiser_.fit(block).transform(block))
            filled = self.standardiser_.inverse_transform(
                self.numeric_imputer_.transform(self.standardiser_.transform(block))
            )
        filled = np.asarray(filled, dtype=float)
        # log1p is undefined below -1, so negatives go to 0 first.
        where = [self.numeric_.index(column) for column in self.recipe["log1p"]]
        filled[:, where] = np.log1p(np.clip(filled[:, where], 0, None))
        return filled

    def fit(self, X, y=None):
        """Learn every step of the recipe from these rows."""
        self.categorical_ = list(self.recipe["categorical"])
        self.numeric_ = list(self.recipe["numeric"])
        missing = set(self.categorical_ + self.numeric_).difference(X.columns)
        if missing:
            raise ValueError(f"recipe columns missing from frame: {sorted(missing)}")

        fill = self.recipe["fill"]
        imputers = {
            "median": SimpleImputer(strategy="median"),
            "mean": SimpleImputer(strategy="mean"),
            "zero": SimpleImputer(strategy="constant", fill_value=0),
            "KNN": KNNImputer(n_neighbors=fill.get("n_neighbors", 5)),
            "iterative": IterativeImputer(max_iter=10,
                                          random_state=fill.get("random_state")),
            "mode": SimpleImputer(strategy="most_frequent"),
            "own level": SimpleImputer(strategy="constant", fill_value="(missing)"),
            "proportional": ProportionalImputer(fill.get("random_state")),
        }
        self.numeric_imputer_ = imputers[fill["numeric"]]
        self.categorical_imputer_ = imputers[fill["categorical"]]
        self.standardiser_ = StandardScaler() if fill.get("standardised") else None

        encoding = self.recipe["encoding"]
        self.encoder_ = clone(ENCODERS[encoding]) if self.categorical_ else None
        if encoding == "one-hot" and self.encoder_ is not None:
            self.encoder_.set_params(drop=self.recipe["drop"])
        if self.encoder_ is not None:
            block = self._categorical_block(X, fit=True)
            self.encoder_.fit(block, y)  # only target encoding reads y
        make = SCALERS[self.recipe["scaler"]]
        self.scaler_ = make() if make is not None and self.numeric_ else None
        numeric_block = self._numeric_block(X, fit=True)
        if self.scaler_ is not None:
            self.scaler_.fit(numeric_block)
        # An encoding other than one-hot is a number, so it is scaled too.
        self.label_scaler_ = None
        if make is not None and self.encoder_ is not None and encoding != "one-hot":
            self.label_scaler_ = make().fit(
                np.asarray(self.encoder_.transform(block), dtype=float)
            )
        labels = (self.encoder_.get_feature_names_out(self.categorical_)
                  if encoding == "one-hot" else self.categorical_)
        self.encoded_names_ = np.asarray([*labels, *self.numeric_], dtype=object)
        self.selection_ = self._fit_selection(X, y)
        self.support_ = (
            np.ones(len(self.encoded_names_), dtype=bool)
            if self.selection_ is None else np.asarray(self.selection_.get_support())
        )
        return self

    def _encoded(self, frame):
        """The encoded, transformed and scaled matrix, before selection."""
        labels = np.asarray(
            self.encoder_.transform(self._categorical_block(frame)), dtype=float
        ) if self.encoder_ is not None else np.empty((len(frame), 0))
        if self.label_scaler_ is not None:
            labels = self.label_scaler_.transform(labels)
        numbers = self._numeric_block(frame)
        if self.scaler_ is not None:
            numbers = self.scaler_.transform(numbers)
        return np.hstack([labels, numbers])

    def _fit_selection(self, X, y):
        """Build the recipe's selection rule and fit it on these rows."""
        spec = (self.recipe["strategy"] if self.selection == "inherit"
                else self.selection)
        if spec is None:
            return None
        if y is None:
            raise ValueError("the selection reads the target: pass y to fit")
        width = len(self.encoded_names_) - len(self.numeric_)
        names = partial(_selector_class,
                        scores=partial(effect_scores, categorical_width=width))
        if spec["kind"] == "vote":
            selector = Vote({name: _rebuild(text, names)
                             for name, text in spec["members"].items()},
                            spec["minimum"])
        elif spec["kind"] == "sequence":
            selector = Sequence(*[(stage["stage"], _rebuild(stage["selector"], names))
                                  for stage in spec["stages"]])
        else:
            raise ValueError(f"unknown strategy kind: {spec['kind']!r}")
        frame = pd.DataFrame(self._encoded(X), columns=self.encoded_names_)
        fitted = selector.fit(frame, y)
        if not np.asarray(fitted.get_support()).any():
            raise ValueError("the selection kept no columns on these rows")
        return fitted

    def transform(self, X):
        """Transform rows with the fitted recipe."""
        check_is_fitted(self, "encoder_")
        return self._encoded(X)[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        """The names of the columns `transform` returns, in order."""
        check_is_fitted(self, "encoder_")
        return self.encoded_names_[self.support_]


# -------------------------------------------------------------------------
# The selection rule the log carries
# -------------------------------------------------------------------------

def effect_scores(matrix, outcome, categorical_width=0):
    """Relevance of each encoded column to the outcome, in [0, 1].

    Against a 0/1 outcome, a 0/1 column among the first `categorical_width`
    takes Cramer's V and any other column |point-biserial r|; against any
    other outcome, every column takes |Pearson r|. A constant column scores 0.
    """
    matrix = np.asarray(matrix, dtype=float)
    if not np.isin(np.asarray(outcome), [0, 1]).all():
        values = np.asarray(outcome, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = np.abs([pearsonr(matrix[:, j], values).statistic
                             for j in range(matrix.shape[1])])
        return np.nan_to_num(scores)
    target = np.asarray(outcome, dtype=int)
    scores = []
    for position in range(matrix.shape[1]):
        values = matrix[:, position]
        if position < categorical_width and np.isin(values, [0.0, 1.0]).all():
            table = pd.crosstab(pd.Series(values), pd.Series(target))
            if min(table.shape) < 2:
                scores.append(0.0)
                continue
            statistic = chi2_contingency(table, correction=False)[0]
            scores.append(np.sqrt(
                statistic / (table.to_numpy().sum() * (min(table.shape) - 1))))
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                scores.append(abs(pointbiserialr(target, values).statistic))
    return np.nan_to_num(np.asarray(scores, dtype=float))

class CorrelationFilter:
    """Keep up to `k` columns by relevance, skipping any that correlates above
    `redundancy` with a column already kept; None removes either limit."""

    def __init__(self, k, redundancy, scores):
        """`scores(matrix, outcome)` returns each column's relevance."""
        self.k = k
        self.redundancy = redundancy
        self.scores = scores

    def fit(self, frame, outcome):
        """Choose the columns on these rows."""
        matrix = frame.to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            relevance = self.scores(matrix, outcome)
            between = np.nan_to_num(np.abs(np.corrcoef(matrix, rowvar=False)))
        keep = []
        for candidate in np.argsort(relevance)[::-1]:
            if len(keep) == self.k:
                break
            if self.redundancy is not None and any(
                between[candidate, chosen] > self.redundancy for chosen in keep
            ):
                continue
            keep.append(int(candidate))
        self.support_ = np.zeros(matrix.shape[1], dtype=bool)
        self.support_[keep] = True
        return self

    def get_support(self):
        """The mask of kept columns."""
        return self.support_

class Sequence:
    """Apply selectors in turn, each to the columns the previous one kept."""

    def __init__(self, *stages):
        """Store the (name, selector) stages in order."""
        self.stages = stages

    def fit(self, frame, outcome):
        """Fit each stage on the columns still kept."""
        support = np.ones(frame.shape[1], dtype=bool)
        for _, stage in self.stages:
            kept = stage.fit(frame.loc[:, support], outcome).get_support()
            support[np.flatnonzero(support)] = kept
        self.support_ = support
        return self

    def get_support(self):
        """The mask of kept columns."""
        return self.support_

class Vote:
    """Keep the columns at least `minimum` of the members select."""

    def __init__(self, members, minimum):
        """Store the named members and the votes a column needs."""
        self.members = members
        self.minimum = minimum

    def fit(self, frame, outcome):
        """Fit every member on these rows and count the votes."""
        votes = sum(np.asarray(member.fit(frame, outcome).get_support(), dtype=int)
                    for member in self.members.values())
        self.support_ = votes >= self.minimum
        return self

    def get_support(self):
        """The mask of kept columns."""
        return self.support_

# Libraries searched for a selector the recipe names.
SELECTION_LIBRARIES = (
    "sklearn.feature_selection",
    "sklearn.linear_model",
    "sklearn.ensemble",
    "sklearn.tree",
    "sklearn.svm",
    "sklearn.neighbors",
)

def _selector_class(name, scores):
    """Return the selector class a recipe names."""
    if name == "CorrelationFilter":
        return partial(CorrelationFilter, scores=scores)
    for library in SELECTION_LIBRARIES:
        found = getattr(importlib.import_module(library), name, None)
        if found is not None:
            return found
    raise ValueError(f"{name} is not defined here nor in {', '.join(SELECTION_LIBRARIES)}")


# -------------------------------------------------------------------------
# Fitting a recipe and a model together, and searching over candidates
# -------------------------------------------------------------------------

from tempfile import TemporaryDirectory
from joblib import Memory
from sklearn.base import is_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    ParameterGrid, RepeatedKFold, RepeatedStratifiedKFold, check_cv)
from sklearn.utils import get_tags
from sklearn.utils.metaestimators import available_if
from sklearn.utils.parallel import Parallel, delayed


OUTER_SPLITS = 5
OUTER_REPEATS = 2


def outer_splitter(task, *, random_state):
    """Return a repeated K-fold splitter, stratified for classification.

    `task` is "classification" or "regression" and picks
    RepeatedStratifiedKFold or RepeatedKFold. `random_state` seeds the fold
    assignment, so two calls with the same task and seed yield the same folds
    and the scores they produce are paired.
    """
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be classification or regression, not {task!r}")
    design = RepeatedStratifiedKFold if task == "classification" else RepeatedKFold
    return design(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=random_state)


def outer_folds(splitter, X, y):
    """Return the row labels each fold scores, as one list per fold.

    Labels come from `X.index` when it has one and are positions otherwise. Two
    runs that produce equal lists divided the same rows the same way.
    """
    return [X.index[score_at].tolist() if hasattr(X, "index") else list(score_at)
            for _, score_at in splitter.split(X, y)]


def _fit_preprocessor(preprocessor, X, y):
    """Fit one cloned preprocessor; cache keys include the rows, labels and recipe."""
    fitted = clone(preprocessor)
    return fitted, fitted.fit_transform(X, y)


def _transform_preprocessor(preprocessor, X):
    """Transform rows with one fitted recipe; its fitted state is part of the cache key."""
    return preprocessor.transform(X)


class PreparedEstimator(BaseEstimator):
    """Fit preprocessing and a model, optionally calibrating their complete fit.

    Calibration splits raw rows before fitting either part. A search may supply
    a Memory cache to reuse preprocessing only for identical training inputs.
    """

    def __init__(self, preprocessor, estimator, *, calibration=None, memory=None):
        """Store the unfitted parts, optional calibration method and fit cache."""
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.calibration = calibration
        self.memory = memory

    def __sklearn_tags__(self):
        """The estimator's type and target tags."""
        tags = super().__sklearn_tags__()
        wrapped = get_tags(self.estimator)
        tags.estimator_type = wrapped.estimator_type
        tags.target_tags = wrapped.target_tags
        tags.classifier_tags = wrapped.classifier_tags
        tags.regressor_tags = wrapped.regressor_tags
        return tags

    def fit(self, X, y):
        """Fit on raw rows, keeping calibration validation rows out of preprocessing."""
        if self.calibration is not None:
            base = clone(self).set_params(calibration=None)
            self.estimator_ = CalibratedClassifierCV(
                base, method=self.calibration).fit(X, y)
        else:
            fit_preprocessor = (_fit_preprocessor if self.memory is None
                                else self.memory.cache(_fit_preprocessor))
            self.preprocessor_, ready = fit_preprocessor(self.preprocessor, X, y)
            self.estimator_ = clone(self.estimator).fit(ready, y)
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        return self

    def _model_input(self, X):
        """Pass raw rows to calibration and prepared rows to an ordinary model."""
        if self.calibration is not None:
            return X
        transform = (_transform_preprocessor if self.memory is None
                     else self.memory.cache(_transform_preprocessor))
        return transform(self.preprocessor_, X)

    def predict(self, X):
        """Predict through the fitted preprocessing and model."""
        return self.estimator_.predict(self._model_input(X))

    @available_if(lambda self: self.calibration is not None
                  or hasattr(self.estimator, "predict_proba"))
    def predict_proba(self, X):
        """Class probabilities through the complete fitted estimator."""
        return self.estimator_.predict_proba(self._model_input(X))

    @available_if(lambda self: self.calibration is None
                  and hasattr(self.estimator, "decision_function"))
    def decision_function(self, X):
        """Decision scores when the uncalibrated estimator supports them."""
        return self.estimator_.decision_function(self._model_input(X))


def _prepare_fold(preprocessor, X, y, fit_at, score_at):
    """Fit a copy of `preprocessor` on one fold's fit rows and transform both sides."""
    X_fit, y_fit = X.iloc[fit_at], y.iloc[fit_at]
    fitted = clone(preprocessor).fit(X_fit, y_fit)
    return {"X_fit": fitted.transform(X_fit), "y_fit": y_fit,
            "X_score": fitted.transform(X.iloc[score_at]), "y_score": y.iloc[score_at]}


def prepared_folds(preprocessor, folds, X, y, *, n_jobs=None):
    """One dict per fold: 'X_fit', 'y_fit', 'X_score' and 'y_score'.

    `folds` is a splitter or a list of (fit, score) positions; a copy of
    `preprocessor` is fitted on each fold's fit rows alone.
    """
    splits = folds.split(X, y) if hasattr(folds, "split") else folds
    completed = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_prepare_fold)(preprocessor, X, y, fit_at, score_at)
        for fit_at, score_at in splits)
    return list(tqdm(completed, total=folds.get_n_splits(X, y) if hasattr(folds, "get_n_splits") else None,
                     desc="Prepare training folds", unit="fold", leave=False))


# -------------------------------------------------------------------------
# A search that never prepares the same rows twice, and a uniform draw
# -------------------------------------------------------------------------

import contextlib
import copy
import os
import uuid
from tempfile import TemporaryDirectory

import joblib
from joblib import Memory
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import (
    GridSearchCV, ParameterGrid, ParameterSampler, RandomizedSearchCV, check_cv)
from sklearn.utils import check_random_state, get_tags, indexable
from sklearn.utils.metaestimators import available_if
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted

try:
    from filelock import FileLock
except ImportError:  # without it a search may repeat a fit two workers start together, never change a number
    FileLock = None


class LockedMemory(Memory):
    """A joblib `Memory` that computes each cache entry once, even when workers ask together.

    `cache(func)` stores one file per distinct call, named by a hash of the
    function's bytecode, its name and its arguments (those listed in `ignore`
    left out). A worker that finds the file reads it. A worker that does not
    takes a lock named by the same hash, looks again, and only then computes
    and stores the result, so two workers never compute the same entry. Unlike
    `Memory.cache`, it never reads the function's source file, so a worker that
    cannot open that file does not invalidate the cache.
    """

    def cache(self, func=None, ignore=None, *args, **kwargs):
        """Cache `func` in `location`, guarding each entry with a lock."""
        if func is None or self.location is None:
            return super().cache(func, ignore, *args, **kwargs)
        folder = os.path.join(self.location, "locked")
        os.makedirs(folder, exist_ok=True)
        code = func.__code__
        identity = (func.__module__, func.__qualname__, code.co_code, repr(code.co_consts), code.co_names)
        left_out = set(ignore or ())

        def cached(*call_args, **call_kwargs):
            kept = {name: value for name, value in call_kwargs.items() if name not in left_out}
            key = joblib.hash((identity, call_args, kept))
            path = os.path.join(folder, key + ".pkl")
            if os.path.exists(path):
                return joblib.load(path)
            with (FileLock(os.path.join(folder, key + ".lock")) if FileLock is not None else contextlib.nullcontext()):
                if os.path.exists(path):
                    return joblib.load(path)
                result = func(*call_args, **call_kwargs)
                partial = path + "." + uuid.uuid4().hex + ".tmp"
                joblib.dump(result, partial)
                os.replace(partial, path)
                return result

        return cached


def _search_rows(values, positions):
    """The rows at `positions` of a pandas object or an array."""
    return values.iloc[positions] if hasattr(values, "iloc") else values[positions]


def _same_folds_every_call(cv):
    """True when `cv.split` returns the same folds each time it is called."""
    seed = getattr(cv, "random_state", 0)
    if isinstance(seed, np.random.RandomState):
        return False
    return seed is not None or getattr(cv, "shuffle", True) is False


def _prepare_rows(memory, preprocessor, X_fit, y_fit, X_score, X_extra):
    """Fill the cache for one setting and one split: the fit, then the transform of each row set."""
    fit = memory.cache(_fit_preprocessor)
    fit(preprocessor, X_fit, y_fit)
    fitted, _ = fit(preprocessor, X_fit, y_fit)  # the stored copy, which is what a later transform is keyed by
    transform = memory.cache(_transform_preprocessor)
    for rows in (X_score, X_extra):
        if rows is not None:
            transform(fitted, rows)


def _search_has(name):
    """`available_if` check: the wrapped search, fitted or not, offers `name`."""
    def check(self):
        return hasattr(self.search_ if hasattr(self, "search_") else self.search, name)
    return check


class CachedSearchCV(BaseEstimator):
    """Run any scikit-learn search so that no preprocessing is repeated.

    `search` is a search object such as `GridSearchCV`, `RandomizedSearchCV` or a
    halving search, whose estimator is a `PreparedEstimator`. During `fit` the
    estimator receives a temporary disk cache keyed by the preprocessor settings and the
    rows, the search runs, and the cache is removed. After the fit the wrapper
    holds the search's results and answers `predict`, `predict_proba`, `score` and
    the other methods the best estimator has.

    A grid, or a random search with an integer `random_state`, lists its
    candidates before it runs, so every distinct setting is fitted once per
    fold first, in parallel, and every fit inside the search is a cache hit. A
    search that picks candidates as it goes fills the cache while it runs, and a
    lock makes two workers wait for one fit instead of repeating it. A search whose
    estimator has no free `memory` parameter runs exactly as it would unwrapped.
    """

    def __init__(self, search):
        """Store the search to run."""
        self.search = search

    def __sklearn_tags__(self):
        """The wrapped search's tags."""
        return copy.deepcopy(get_tags(self.search))

    def __sklearn_is_fitted__(self):
        """Fitted once the search has run."""
        return hasattr(self, "search_")

    def _candidates(self, search):
        """The candidates a search will score, or None when it chooses them as it goes."""
        if isinstance(search, GridSearchCV):
            return list(ParameterGrid(search.param_grid))
        if isinstance(search, RandomizedSearchCV) and isinstance(search.random_state, (int, np.integer)):
            return list(ParameterSampler(search.param_distributions, search.n_iter, random_state=search.random_state))
        return None

    def _prewarm(self, search, memory, X, y, groups):
        """Fit every distinct (preprocessor setting, fit rows) pair once, before the search runs."""
        candidates = self._candidates(search)
        base = search.estimator
        cv = check_cv(search.cv, y, classifier=is_classifier(base))
        if candidates is None or not isinstance(base, PreparedEstimator) or not _same_folds_every_call(cv):
            return
        X, y, groups = indexable(X, y, groups)
        settings = {}
        for candidate in candidates:
            model = clone(base).set_params(**candidate)
            if getattr(model, "calibration", None) is None:
                settings.setdefault(joblib.hash(model.preprocessor), model.preprocessor)
        splits = list(cv.split(X, y, groups))
        extra = bool(getattr(search, "return_train_score", False))
        Parallel(n_jobs=search.n_jobs)(
            delayed(_prepare_rows)(memory, preprocessor, _search_rows(X, fit_at), _search_rows(y, fit_at),
                                   _search_rows(X, score_at), _search_rows(X, fit_at) if extra else None)
            for preprocessor in settings.values() for fit_at, score_at in splits)

    def fit(self, X, y=None, **params):
        """Run the search with a temporary preparation cache and keep its results."""
        search = clone(self.search)
        estimator = getattr(search, "estimator", None)
        cached = (estimator is not None and hasattr(estimator, "get_params")
                  and "memory" in estimator.get_params(deep=False) and estimator.get_params(deep=False)["memory"] is None)
        with TemporaryDirectory(prefix="prepared-search-") as location:
            if cached:
                memory = LockedMemory(location, verbose=0)
                search.set_params(estimator__memory=memory)
                self._prewarm(search, memory, X, y, params.get("groups"))
            search.fit(X, y, **params)
            if cached:
                # The folder is removed on leaving: no fitted model may keep a handle on it.
                search.set_params(estimator__memory=None)
                if hasattr(search, "best_estimator_"):
                    search.best_estimator_.set_params(memory=None)
        self.search_ = search
        for name, value in vars(search).items():
            if name.endswith("_") and not name.startswith("_") and not isinstance(getattr(type(self), name, None), property):
                setattr(self, name, value)
        return self

    def _fitted_search(self):
        """The fitted search; calling before `fit` raises a `NotFittedError`."""
        check_is_fitted(self, "search_")
        return self.search_

    @property
    def classes_(self):
        """The class labels of the best estimator."""
        return self._fitted_search().classes_

    @property
    def n_features_in_(self):
        """The number of columns seen by the best estimator."""
        return self._fitted_search().n_features_in_

    @property
    def feature_names_in_(self):
        """The column names seen by the best estimator."""
        return self._fitted_search().feature_names_in_

    @available_if(_search_has("predict"))
    def predict(self, X):
        """Predict with the best estimator."""
        return self._fitted_search().predict(X)

    @available_if(_search_has("predict_proba"))
    def predict_proba(self, X):
        """Class probabilities from the best estimator."""
        return self._fitted_search().predict_proba(X)

    @available_if(_search_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Log class probabilities from the best estimator."""
        return self._fitted_search().predict_log_proba(X)

    @available_if(_search_has("decision_function"))
    def decision_function(self, X):
        """Decision scores from the best estimator."""
        return self._fitted_search().decision_function(X)

    @available_if(_search_has("transform"))
    def transform(self, X):
        """Transform rows with the best estimator."""
        return self._fitted_search().transform(X)

    @available_if(_search_has("inverse_transform"))
    def inverse_transform(self, X):
        """Invert the best estimator's transform."""
        return self._fitted_search().inverse_transform(X)

    @available_if(_search_has("score_samples"))
    def score_samples(self, X):
        """Per-row scores from the best estimator."""
        return self._fitted_search().score_samples(X)

    def score(self, X, y=None, **params):
        """Score with the search's scorer, as the search does."""
        return self._fitted_search().score(X, y, **params)


# -- Week 11: raw-row ownership for learned combinations ----------------------
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.base import is_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import get_tags
import copy
import joblib


def constructor_value(value):
    """Display predictive settings while fixed raw-row preparation stays explicit separately."""
    if isinstance(value, PreparedEstimator):
        base = constructor_value(value.estimator)
        return ("CalibratedClassifierCV(estimator=" + base + ", method=" + repr(value.calibration) + ")"
                if value.calibration is not None else base)
    if hasattr(value, "get_params"):
        return type(value).__name__ + "(" + constructor_arguments(value) + ")"
    if value is np.log:
        return "np.log"
    if value is np.exp:
        return "np.exp"
    if isinstance(value, (list, tuple)):
        left, right = ("[", "]") if isinstance(value, list) else ("(", ")")
        return left + ", ".join(constructor_value(item) for item in value) + ("," if isinstance(value, tuple) and len(value) == 1 else "") + right
    return repr(value)


def constructor_arguments(model, exclude=()):
    controls = {"memory", "random_state", "n_jobs", "verbose", "warm_start"}
    return ", ".join(name + "=" + ("preparation" if name == "preprocessor" else constructor_value(value))
                     for name, value in model.get_params(deep=False).items()
                     if name not in controls and name not in exclude)


class AdaRawFrame(TransformerMixin, BaseEstimator):
    """Make missing raw values finite for AdaBoost without learning any statistics."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(object).where(pd.notna(X), "__w11_missing__").to_numpy()


class SampleWeightedPreparedEstimator(PreparedEstimator):
    """An AdaBoost base learner that keeps raw-row calibration ownership."""
    def _raw(self, X):
        raw = X.copy() if hasattr(X, "columns") else pd.DataFrame(X, columns=self.preprocessor.recipe["categorical"] + self.preprocessor.recipe["numeric"])
        raw = raw.replace("__w11_missing__", np.nan)
        for column in self.preprocessor.recipe["numeric"]:
            raw[column] = pd.to_numeric(raw[column])
        return raw

    def fit(self, X, y, sample_weight=None):
        raw = self._raw(X)
        if self.calibration is not None:
            base = clone(self).set_params(calibration=None)
            self.estimator_ = CalibratedClassifierCV(base, method=self.calibration).fit(raw, y, sample_weight=sample_weight)
        else:
            prepare = _fit_preprocessor if self.memory is None else self.memory.cache(_fit_preprocessor)
            self.preprocessor_, ready = prepare(self.preprocessor, raw, y)
            self.estimator_ = (clone(self.estimator).fit(ready, y) if sample_weight is None
                               else clone(self.estimator).fit(ready, y, sample_weight=sample_weight))
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        return self

    def _model_input(self, X):
        raw = self._raw(X)
        return raw if self.calibration is not None else self.preprocessor_.transform(raw)


def weighted_ada_members(preprocessor, estimators, memory=None):
    """Keep each eligible taught constructor and put calibration around raw rows."""
    members = {}
    for name, estimator in estimators.items():
        if isinstance(estimator, CalibratedClassifierCV):
            members[name] = SampleWeightedPreparedEstimator(clone(preprocessor), clone(estimator.estimator), calibration=estimator.method, memory=memory)
        else:
            members[name] = SampleWeightedPreparedEstimator(clone(preprocessor), clone(estimator), memory=memory)
    return members


_FITS = {}


def _complete_fit(preprocessor, estimator, calibration, X, y, memory=None):
    """One complete member fit per distinct rows; the latest few stay in memory for reuse."""
    key = joblib.hash((preprocessor, estimator, calibration, X, y))
    if key not in _FITS:
        if len(_FITS) >= 12:
            _FITS.pop(next(iter(_FITS)))
        _FITS[key] = PreparedEstimator(preprocessor, estimator, calibration=calibration, memory=memory).fit(X, y)
    return _FITS[key]


def _member_output(preprocessor, estimator, calibration, X, y, X_score, method, memory=None):
    return getattr(_complete_fit(preprocessor, estimator, calibration, X, y, memory), method)(X_score)


class CachedPreparedMember(PreparedEstimator):
    """A member that fits lazily; predictions for identical fit rows and score rows are computed once.

    Preprocessing is shared by every member that sees the same fit rows. Only a
    member whose randomness is fixed is reused, so a cached answer equals a fresh one.
    """
    def fit(self, X, y):
        unfixed_randomness = any(name.endswith("random_state") and value is None for name, value in self.estimator.get_params(deep=True).items())
        if self.memory is None or unfixed_randomness:
            return super().fit(X, y)
        self.X_, self.y_ = X, y
        if self.calibration is not None or is_classifier(self.estimator):
            self.classes_ = np.unique(y)
        return self

    def _model_input(self, X):
        return X if self.calibration is not None else self.preprocessor_.transform(X)

    def _output(self, X, method):
        if not hasattr(self, "X_"):
            return getattr(super(), method)(X)
        cached = self.memory.cache(_member_output, ignore=["memory"])
        return cached(self.preprocessor, self.estimator, self.calibration, self.X_, self.y_, X, method, memory=self.memory)

    def predict(self, X):
        return self._output(X, "predict")

    @available_if(lambda self: self.calibration is not None or hasattr(self.estimator, "predict_proba"))
    def predict_proba(self, X):
        return self._output(X, "predict_proba")


def fold_owned_members(preprocessor, estimators, memory=None):
    """Unfitted members whose calibration also owns its raw training split.

    A boosted learner whose base already prepares raw rows takes the raw frame instead.
    """
    members = []
    for name, estimator in estimators:
        if isinstance(estimator, CalibratedClassifierCV):
            model = CachedPreparedMember(clone(preprocessor), clone(estimator.estimator), calibration=estimator.method, memory=memory)
        elif isinstance(getattr(estimator, "estimator", None), SampleWeightedPreparedEstimator):
            model = CachedPreparedMember(AdaRawFrame(), clone(estimator), memory=memory)
        else:
            model = CachedPreparedMember(clone(preprocessor), clone(estimator), memory=memory)
        members.append((name, model))
    return members


def unprepared_boosting(estimator):
    """A boosted learner that learns from prediction columns: its prepared base becomes the plain base."""
    base = getattr(estimator, "estimator", None)
    if not isinstance(base, SampleWeightedPreparedEstimator):
        return clone(estimator)
    inner = clone(base.estimator)
    if base.calibration is not None:
        inner = CalibratedClassifierCV(inner, method=base.calibration)
    return clone(estimator).set_params(estimator=inner)


class FoldStacking(BaseEstimator):
    """Cross-fit raw-row members, then fit a final learner on their predictions."""
    def __init__(self, preprocessor, estimators, final_estimator, *, cv=3, method="auto", passthrough=False, random_state=33, memory=None):
        self.preprocessor = preprocessor
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.method = method
        self.passthrough = passthrough
        self.random_state = random_state
        self.memory = memory

    def __sklearn_tags__(self):
        return copy.deepcopy(get_tags(self.estimators[0][1]))

    def fit(self, X, y):
        classification = is_classifier(self.estimators[0][1])
        self.method_ = ("predict_proba" if classification else "predict") if self.method == "auto" else self.method
        splitter = (StratifiedKFold if classification else KFold)(self.cv, shuffle=True, random_state=self.random_state)
        members = fold_owned_members(self.preprocessor, self.estimators, memory=self.memory)
        meta = np.empty((len(y), len(members)))
        self.ownership_ = []
        for fit_at, score_at in splitter.split(X, y):
            assert not np.intersect1d(fit_at, score_at).size
            self.ownership_.append((X.index[fit_at].tolist(), X.index[score_at].tolist()))
            for column, (_, member) in enumerate(members):
                fitted = clone(member).fit(X.iloc[fit_at], y.iloc[fit_at])
                prediction = getattr(fitted, self.method_)(X.iloc[score_at])
                meta[score_at, column] = prediction[:, 1] if prediction.ndim == 2 else prediction
        # The final learner owns one stable prepared-feature schema. Base
        # predictions above still use their separate inner-fold preparation.
        if hasattr(self, "passthrough_preprocessor_"):
            del self.passthrough_preprocessor_
        if self.passthrough:
            self.passthrough_preprocessor_ = clone(self.preprocessor)
            ready = self.passthrough_preprocessor_.fit_transform(X, y)
            meta = np.column_stack((meta, ready))
        self.meta_estimator_ = clone(self.final_estimator).fit(meta, y)
        self.estimators_ = [(name, clone(member).fit(X, y)) for name, member in members]
        if classification:
            self.classes_ = self.meta_estimator_.classes_
        return self

    def _meta(self, X):
        columns = []
        for _, member in self.estimators_:
            prediction = getattr(member, self.method_)(X)
            columns.append(prediction[:, 1] if prediction.ndim == 2 else prediction)
        meta = np.column_stack(columns)
        if getattr(self, "passthrough", False):
            meta = np.column_stack((meta, self.passthrough_preprocessor_.transform(X)))
        return meta

    def predict(self, X):
        return self.meta_estimator_.predict(self._meta(X))

    @available_if(lambda self: is_classifier(self.estimators[0][1]))
    def predict_proba(self, X):
        return self.meta_estimator_.predict_proba(self._meta(X))


class FoldBlending(FoldStacking):
    """Hold out raw training rows for the meta-learner; never refit members on them."""
    def __init__(self, preprocessor, estimators, final_estimator, *, fraction=0.25, method="auto", random_state=33, memory=None):
        self.preprocessor = preprocessor
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.fraction = fraction
        self.method = method
        self.random_state = random_state
        self.memory = memory

    def fit(self, X, y):
        classification = is_classifier(self.estimators[0][1])
        self.method_ = ("predict_proba" if classification else "predict") if self.method == "auto" else self.method
        fit_at, score_at = train_test_split(np.arange(len(y)), test_size=self.fraction,
            stratify=y if classification else None, random_state=self.random_state)
        self.ownership_ = [(X.index[fit_at].tolist(), X.index[score_at].tolist())]
        members = fold_owned_members(self.preprocessor, self.estimators, memory=self.memory)
        self.estimators_ = [(name, clone(member).fit(X.iloc[fit_at], y.iloc[fit_at])) for name, member in members]
        meta = self._meta(X.iloc[score_at])
        self.meta_estimator_ = clone(self.final_estimator).fit(meta, y.iloc[score_at])
        if classification:
            self.classes_ = self.meta_estimator_.classes_
        return self


def save_boosting_selection(path, payload, model):
    """Save the selected settings and scores, with the fitted winner alongside."""
    import hashlib
    import sklearn
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_path = path.with_name(f"week_11_{payload['task']}_model.joblib")
    joblib.dump(model, model_path)
    document = dict(payload, schema_version=1, sklearn_version=sklearn.__version__,
                    model_file=model_path.name,
                    model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest())
    joblib.dump(document, path)


def load_boosting_selection(path, *, task, context, fold_fingerprint, n_splits):
    """Reject missing or stale boosting evidence before fitting a combiner."""
    import hashlib
    import sklearn
    path = Path(path)
    instruction = f"Run the Week 11 boosting {task} notebook first, then rerun this notebook."
    required = {"schema_version", "sklearn_version", "task", "context", "fold_fingerprint",
                "boost_settings", "boost_fold_settings", "boost_nested_scores",
                "boost_nested_validation", "boost_nested_train", "boost_train", "boost_validation",
                "boost_label", "boost_test", "baseline_cv", "baseline_test", "tuning_comparison", "model_file", "model_sha256"}
    try:
        document = joblib.load(path)
        if not isinstance(document, dict) or not required <= document.keys():
            raise ValueError("incomplete schema")
        if document["schema_version"] != 1 or document["task"] != task:
            raise ValueError("wrong schema or task")
        comparison = document["tuning_comparison"]
        if not isinstance(comparison, dict) or comparison.get("protocol") != "fixed_settings_on_course_folds":
            raise ValueError("invalid fixed-setting comparison")
        if document["sklearn_version"] != sklearn.__version__:
            raise ValueError("different scikit-learn version")
        if document["context"] != context or document["fold_fingerprint"] != fold_fingerprint:
            raise ValueError("data, recipe, split or folds changed")
        if not isinstance(document["boost_settings"], dict) or len(document["boost_fold_settings"]) != n_splits:
            raise ValueError("invalid selected settings")
        if not all(isinstance(settings, dict) for settings in document["boost_fold_settings"]):
            raise ValueError("invalid outer-fold settings")
        for key in ("boost_nested_validation", "boost_nested_train", "boost_train", "boost_validation"):
            values = np.asarray(document[key], dtype=float)
            if values.shape != (n_splits,) or not np.isfinite(values).all():
                raise ValueError(f"invalid {key}")
        if not isinstance(document["boost_nested_scores"], dict):
            raise ValueError("invalid companion scores")
        for values in document["boost_nested_scores"].values():
            if np.asarray(values).shape != (n_splits,):
                raise ValueError("invalid companion score shape")
        model_path = path.with_name(f"week_11_{task}_model.joblib")
        if document["model_file"] != model_path.name or hashlib.sha256(model_path.read_bytes()).hexdigest() != document["model_sha256"]:
            raise ValueError("fitted winner missing or changed")
        document["boosting_winner"] = joblib.load(model_path)
        return document
    except Exception as exc:
        raise ValueError(f"Boosting selection is missing or stale ({exc}). {instruction}") from exc
