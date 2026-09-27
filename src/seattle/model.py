"""Local Ridge regression, CatBoost regression, and tuning for Seattle sales."""

from functools import partial
from typing import Self, cast

import optuna
from catboost import CatBoostRegressor
from geopandas import GeoDataFrame
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.utils.validation import check_is_fitted

from .house_features import HouseFeatures
from .neighboor_features import LocalRidgeFeatures
from .neighbors import Floats


class _SeattleLinReg(RegressorMixin, BaseEstimator):
    def __init__(self, ridge: LocalRidgeFeatures) -> None:
        self.ridge = ridge

    def fit(self, x: DataFrame, y: Series) -> Self:
        self.ridge_ = clone(self.ridge).fit(x, y)
        return self

    def _prediction_features(self, x: DataFrame) -> DataFrame:
        check_is_fitted(self, "ridge_")
        return self.ridge_.transform(x)

    def predict(self, x: DataFrame) -> Floats:
        return self._prediction_features(x)["ridge_price"].to_numpy(dtype=float)


def _catboost_columns(linreg: _SeattleLinReg) -> list[str]:
    return [col for col in linreg.ridge_.onehot_cols if col not in {"id", "date", "price"}]


def _catboost_inputs(linreg: _SeattleLinReg, x: DataFrame) -> DataFrame:
    frame = linreg._prediction_features(x)  # noqa: SLF001
    ridge = linreg.ridge_
    inputs = frame[[*ridge.numeric_cols_, *ridge.output_cols_]].copy()
    for col in _catboost_columns(linreg):
        inputs[col] = frame[col].astype("string").fillna("<missing>")
    return inputs


class _SeattleCatBoost(RegressorMixin, BaseEstimator):
    def __init__(self, linreg: _SeattleLinReg, catboost: CatBoostRegressor) -> None:
        self.linreg = linreg
        self.catboost = catboost

    def fit(self, x: DataFrame, y: Series) -> Self:
        """Fit CatBoost on sale prices for houses the linear model has not seen."""
        features = _catboost_inputs(self.linreg, x)
        self.catboost_ = clone(self.catboost).fit(
            features, y.to_numpy(dtype=float), cat_features=_catboost_columns(self.linreg)
        )
        return self

    def predict(self, x: DataFrame) -> Floats:
        check_is_fitted(self, "catboost_")
        features = _catboost_inputs(self.linreg, x)
        return self.catboost_.predict(features)


def build_seattle_linreg(  # noqa: PLR0913
    *,
    water: GeoDataFrame,
    radius_km: float,
    alpha: float = 1.0,
    predict_neighbors: int = 1,
    onehot_cols: tuple[str, ...] = (),
    mean_cols: tuple[str, ...] = ("zipcode",),
    n_jobs: int = -1,
    n_splits: int = 3,
    random_state: int = 42,
    progress: bool = True,
) -> _SeattleLinReg:
    """Build a regressor that averages price predictions from the nearest Ridge stacks."""
    house = HouseFeatures(
        water=water,
        radius_km=radius_km,
        n_splits=n_splits,
        random_state=random_state,
        mean_encode=mean_cols,
    )
    ridge = LocalRidgeFeatures(
        house,
        onehot_cols=onehot_cols,
        mean_cols=mean_cols,
        alpha=alpha,
        predict_neighbors=predict_neighbors,
        n_jobs=n_jobs,
        progress=progress,
    )
    return _SeattleLinReg(ridge)


def build_catboost(
    linreg: _SeattleLinReg,
    *,
    iterations: int | None = None,
    depth: int | None = None,
    learning_rate: float | None = None,
    **catboost_params: object,
) -> _SeattleCatBoost:
    """Use a fitted Ridge model's predictions as features for CatBoost on separate houses."""
    check_is_fitted(linreg, "ridge_")
    params = {"verbose": False, "allow_writing_files": False, **catboost_params}
    return _SeattleCatBoost(
        linreg, CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=learning_rate, **params)
    )


def tune_seattle_linreg(  # noqa: PLR0913
    x: DataFrame,
    y: Series,
    *,
    water: GeoDataFrame,
    n_trials: int,
    search_space: dict[str, tuple[float, float]],
    initial_params: dict[str, float] | None = None,
    onehot_cols: tuple[str, ...] = (),
    mean_cols: tuple[str, ...] = ("zipcode",),
    n_jobs: int = -1,
    n_splits: int = 3,
    random_state: int = 42,
    progress: bool = False,
) -> tuple[optuna.Study, _SeattleLinReg]:
    """Minimize house-grouped validation MAPE and refit the best model on all supplied sales.

    Pass (low, high) pairs in search_space for radius_km and optionally alpha and predict_neighbors.
    Radius and alpha are sampled on a log scale; predict_neighbors uses integer bounds.
    Omitted optional parameters retain their builder defaults.
    Pass initial_params to evaluate a starting point as the first trial.
    """
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(splitter.split(x, groups=x["id"]))
    build = partial(
        build_seattle_linreg,
        water=water,
        onehot_cols=onehot_cols,
        mean_cols=mean_cols,
        n_jobs=n_jobs,
        n_splits=n_splits,
        random_state=random_state,
        progress=progress,
    )

    def objective(trial: optuna.Trial) -> float:
        params = {}
        for name, bounds in search_space.items():
            if name == "predict_neighbors":
                params[name] = trial.suggest_int(name, *cast("tuple[int, int]", bounds))
            else:
                params[name] = trial.suggest_float(name, *bounds, log=True)
        model = build(**params)
        scores = cross_val_score(
            model, x, y, cv=folds, scoring="neg_mean_absolute_percentage_error", n_jobs=1, error_score="raise"
        )
        return -float(scores.mean())

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    if initial_params is not None:
        study.enqueue_trial(initial_params)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=progress)
    return study, build(**study.best_params).fit(x, y)


def tune_seattle_catboost(  # noqa: PLR0913
    linreg: _SeattleLinReg,
    x: DataFrame,
    y: Series,
    *,
    n_trials: int,
    iterations_range: tuple[int, int],
    depth_range: tuple[int, int],
    learning_rate_range: tuple[float, float],
    n_splits: int = 3,
    random_state: int = 42,
    progress: bool = False,
    **catboost_params: object,
) -> tuple[optuna.Study, _SeattleCatBoost]:
    """Tune final-price MAPE on house-grouped folds without refitting the linear model.

    The development data must contain only houses unseen by the fitted linear model.
    The returned CatBoost model is fitted on all development rows.
    """
    features = _catboost_inputs(linreg, x)
    cat_cols = _catboost_columns(linreg)
    prices = y.to_numpy(dtype=float)
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(splitter.split(x, groups=x["id"]))

    def objective(trial: optuna.Trial) -> float:
        params = {
            **catboost_params,
            "iterations": trial.suggest_int("iterations", *iterations_range),
            "depth": trial.suggest_int("depth", *depth_range),
            "learning_rate": trial.suggest_float("learning_rate", *learning_rate_range, log=True),
        }
        template = build_catboost(linreg, **params).catboost
        scores = []
        for train, valid in folds:
            model = clone(template).fit(features.iloc[train], prices[train], cat_features=cat_cols)
            prediction = model.predict(features.iloc[valid])
            scores.append(mean_absolute_percentage_error(prices[valid], prediction))
        return float(sum(scores) / len(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=progress)
    composite = build_catboost(linreg, **(catboost_params | study.best_params))
    composite.catboost_ = clone(composite.catboost).fit(features, prices, cat_features=cat_cols)
    return study, composite
