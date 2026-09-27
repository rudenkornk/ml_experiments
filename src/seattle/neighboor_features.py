"""Two-stage local Ridge features for Seattle sales."""

from typing import Self

import numpy as np
from joblib import Parallel, delayed, parallel_config
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

from .house_features import HouseFeatures
from .neighbors import Floats, Positions

DIFF_COLS = ("grade", "condition", "sqft_living", "sqft_above", "sqft_lot")
DIFF_FEATURES = tuple(f"{col}_vs_neighbor_median" for col in DIFF_COLS)
RATIO_FEATURES = tuple(f"{col}_over_neighbor_median" for col in DIFF_COLS)


class _LocalStack:
    def __init__(self, *, ratio_alpha: float, price_alpha: float) -> None:
        self.ratio_alpha = ratio_alpha
        self.price_model = self._ridge(price_alpha)

    @staticmethod
    def _ridge(alpha: float) -> Pipeline:
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, inputs: Floats, prices: Floats, ratios: Floats, areas: Floats) -> Self:
        self.ratio_models: dict[int, Pipeline] = {}
        area_medians = []
        for i, target in enumerate(ratios.T):
            valid_area = np.isfinite(areas[:, i])
            observed = valid_area & np.isfinite(target)
            if observed.any():
                self.ratio_models[i] = self._ridge(self.ratio_alpha).fit(inputs[observed], target[observed])
                area_medians.append(np.median(areas[valid_area, i]))
        self.area_medians = np.asarray(area_medians, dtype=float)
        self.n_ratios = ratios.shape[1]
        predictions = self._predict_ratios(inputs)
        self.price_model.fit(self._price_inputs(inputs, predictions, areas), prices)
        return self

    def _predict_ratios(self, inputs: Floats) -> Floats:
        ratios = np.full((len(inputs), self.n_ratios), np.nan)
        for i, model in self.ratio_models.items():
            ratios[:, i] = model.predict(inputs)
        return ratios

    def _price_inputs(self, inputs: Floats, ratios: Floats, areas: Floats) -> Floats:
        columns = list(self.ratio_models)
        selected_areas = areas[:, columns]
        filled_areas = np.where(np.isfinite(selected_areas), selected_areas, self.area_medians)
        return np.column_stack((inputs, ratios[:, columns] * filled_areas))

    def predict(self, inputs: Floats, areas: Floats) -> Floats:
        ratios = self._predict_ratios(inputs)
        return np.column_stack((ratios, self.price_model.predict(self._price_inputs(inputs, ratios, areas))))


def _fit_stack(  # noqa: PLR0913
    inputs: Floats, prices: Floats, targets: Floats, areas: Floats, rows: Positions, *, alpha: float
) -> _LocalStack:
    return _LocalStack(ratio_alpha=alpha, price_alpha=alpha).fit(inputs[rows], prices[rows], targets[rows], areas[rows])


class LocalRidgeFeatures(TransformerMixin, BaseEstimator):
    """Append ratio and price estimates from geographically nearby fitted stacks."""

    def __init__(  # noqa: PLR0913
        self,
        house_features: HouseFeatures,
        *,
        onehot_cols: tuple[str, ...] = (),
        mean_cols: tuple[str, ...] = ("zipcode",),
        alpha: float = 1.0,
        predict_neighbors: int = 1,
        n_jobs: int = -1,
        progress: bool = True,
    ) -> None:
        """Configure fold-level encoders and regularization for both Ridge stages."""
        self.house_features = house_features
        self.onehot_cols = onehot_cols
        self.mean_cols = mean_cols
        self.alpha = alpha
        self.predict_neighbors = predict_neighbors
        self.n_jobs = n_jobs
        self.progress = progress

    def _add_neighbor_diffs(self, frame: DataFrame, neighborhoods: list[Positions]) -> DataFrame:
        values = frame[list(DIFF_COLS)].to_numpy(dtype=float, na_value=np.nan)
        medians = np.empty_like(values)
        for i, rows in enumerate(neighborhoods):
            medians[i] = self.reference_values_.iloc[rows].median().to_numpy(dtype=float, na_value=np.nan)
        ratios = np.divide(
            values, medians, out=np.full_like(values, np.nan), where=(medians != 0) & np.isfinite(medians)
        )
        return frame.assign(
            **dict(zip(DIFF_FEATURES, (values - medians).T, strict=True)),
            **dict(zip(RATIO_FEATURES, ratios.T, strict=True)),
        )

    def _fit_reference(self, x: DataFrame, y: Series) -> DataFrame:
        self.features_: HouseFeatures = clone(self.house_features)
        self.features_.set_params(mean_encode=self.mean_cols)
        frame = self.features_.fit_transform(x, y)
        neighborhoods = self.features_.neighborhoods_.rows_
        self.reference_values_ = x[list(DIFF_COLS)].copy()
        frame = self._add_neighbor_diffs(frame, neighborhoods)
        reference = self.features_.reference_prices_
        excluded = {"id", *reference.columns, *self.onehot_cols, *self.mean_cols}
        self.numeric_cols_ = [col for col in frame.select_dtypes(include="number") if col not in excluded]
        self.imputer_ = SimpleImputer(keep_empty_features=True).fit(frame[self.numeric_cols_])
        self.onehot_ = None
        if self.onehot_cols:
            self.onehot_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
                frame[list(self.onehot_cols)]
            )

        inputs = self._inputs(frame)
        prices = reference["price"].to_numpy(dtype=float)
        ratios = reference.drop(columns="price")
        targets = ratios.to_numpy(dtype=float)
        self.area_cols_ = [col.removeprefix("price_per_") for col in ratios]
        areas = self._areas(x)
        self.output_cols_ = [*(f"ridge_{col}" for col in ratios), "ridge_price"]
        with parallel_config(backend="loky", inner_max_num_threads=1):
            models = Parallel(n_jobs=self.n_jobs, return_as="generator")(
                delayed(_fit_stack)(inputs, prices, targets, areas, rows, alpha=self.alpha) for rows in neighborhoods
            )
            self.models_ = list(
                tqdm(
                    models,
                    total=len(neighborhoods),
                    desc="Local Ridge stacks",
                    unit="sale",
                    leave=False,
                    disable=not self.progress,
                )
            )
        return frame

    def _areas(self, x: DataFrame) -> Floats:
        areas = x[self.area_cols_].to_numpy(dtype=float, na_value=np.nan)
        return np.where(areas != 0, areas, np.nan)

    def _inputs(self, frame: DataFrame) -> Floats:
        numeric = np.asarray(self.imputer_.transform(frame[self.numeric_cols_]), dtype=float)
        if self.onehot_ is None:
            return numeric
        return np.column_stack((numeric, self.onehot_.transform(frame[list(self.onehot_cols)])))

    def fit(self, x: DataFrame, y: Series) -> Self:
        """Fit both regression stages for every training sale."""
        self._fit_reference(x, y)
        return self

    def transform(self, x: DataFrame) -> DataFrame:
        """Reuse fitted stacks without retraining on the query neighborhoods."""
        check_is_fitted(self, "models_")
        lookup = self.features_.neighborhoods_
        frame = self.features_.transform(x)
        frame = self._add_neighbor_diffs(frame, lookup.transform(x))
        inputs = self._inputs(frame)
        areas = self._areas(x)
        predictions = np.empty((len(x), len(self.output_cols_)))
        k = min(self.predict_neighbors, len(self.models_))
        for i, anchors in enumerate(lookup.nearest_k(x, k=k)):
            row = slice(i, i + 1)
            nearby = np.vstack([self.models_[anchor].predict(inputs[row], areas[row]) for anchor in anchors])
            predictions[i] = nearby[0]
            predictions[i, -1] = nearby[:, -1].mean()
        frame[self.output_cols_] = predictions
        return frame

    def fit_transform(self, x: DataFrame, y: Series) -> DataFrame:
        """Generate training predictions with all sales of each query house held out."""
        result = self._fit_reference(x, y)
        folds = GroupKFold(n_splits=self.features_.n_splits, shuffle=True, random_state=self.features_.random_state)
        columns = [*DIFF_FEATURES, *RATIO_FEATURES, *self.output_cols_]
        held_out = np.empty((len(x), len(columns)))
        for train, valid in folds.split(x, groups=x["id"]):
            local = clone(self).fit(x.iloc[train], y.iloc[train])
            held_out[valid] = local.transform(x.iloc[valid])[columns].to_numpy()
        result[columns] = held_out
        return result
