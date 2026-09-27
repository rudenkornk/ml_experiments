"""House and local Ridge features for Seattle sales."""

from typing import Self

import geopandas as gpd
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics.pairwise import haversine_distances
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.validation import check_is_fitted

from .neighbors import EARTH_RADIUS_KM, HouseNeighborhoods, Positions

LAKE_CRS = "EPSG:32610"
SALE_DATE_FORMAT = "%Y%m%dT%H%M%S"
CITY_CENTERS = np.deg2rad([[47.6062, -122.3321], [47.6101, -122.2015]])
COMP_AREA_FRACTION = 0.25
COMP_GRADE_GAP = 1


class HouseFeatures(TransformerMixin, BaseEstimator):
    """Add local and encoded features and retain per-sale prices for neighborhood lookup."""

    def __init__(
        self,
        *,
        radius_km: float,
        mean_encode: tuple[str, ...] = ("zipcode",),
        n_splits: int = 3,
        random_state: int = 42,
        water: gpd.GeoDataFrame,
    ) -> None:
        """Configure the geographic and grouped training references."""
        self.radius_km = radius_km
        self.water = water
        self.mean_encode = mean_encode
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, x: DataFrame, y: pd.Series) -> Self:
        """Fit references on the provided training sales."""
        self.neighborhoods_ = HouseNeighborhoods(radius_km=self.radius_km).fit(x)
        self.reference_prices_ = self._prepare_reference_prices(x, y)
        self.comparable_ids_ = x["id"].to_numpy(copy=True)
        self.comparable_attributes_ = x[["sqft_living", "grade"]].to_numpy(dtype=float, na_value=np.nan)
        self.shoreline_ = self.water.to_crs(LAKE_CRS).geometry.union_all().boundary
        self._fit_encoder(x, y)
        return self

    @staticmethod
    def _prepare_reference_prices(x: DataFrame, y: pd.Series) -> DataFrame:
        prices = y.to_numpy(copy=True)
        areas = x[["sqft_living", "sqft_lot", "sqft_above"]].replace(0, np.nan)
        ratios = areas.rdiv(prices, axis="index").add_prefix("price_per_")
        ratios.insert(0, "price", prices)
        return ratios

    def _fit_encoder(self, x: DataFrame, y: pd.Series) -> None:
        self.columns_ = list(self.mean_encode)
        self.encoder_ = None
        if self.columns_:
            self.encoder_ = TargetEncoder(target_type="continuous", smooth="auto")
            self.encoder_.fit(x[self.columns_], y)

    @staticmethod
    def _house_features(x: DataFrame) -> DataFrame:
        date = pd.to_datetime(x["date"], format=SALE_DATE_FORMAT)
        renovated = x["yr_renovated"].gt(0)
        bedrooms = x["bedrooms"].replace(0, np.nan)
        lot = x["sqft_lot"].replace(0, np.nan)
        return x.assign(
            sale_days=(date - pd.Timestamp("2015-01-01")).dt.days,
            age_at_sale=date.dt.year - x["yr_built"],
            years_since_renovation=date.dt.year - x["yr_renovated"].where(renovated),
            log_sqft_living=np.log1p(x["sqft_living"]),
            log_sqft_lot=np.log1p(x["sqft_lot"]),
            living_to_living15=x["sqft_living"] / x["sqft_living15"].replace(0, np.nan),
            lot_to_lot15=lot / x["sqft_lot15"].replace(0, np.nan),
            basement_share=x["sqft_basement"] / x["sqft_living"].replace(0, np.nan),
            living_per_bedroom=x["sqft_living"] / bedrooms,
            bathrooms_per_bedroom=x["bathrooms"] / bedrooms,
            estimated_lot_coverage=x["sqft_above"] / x["floors"].replace(0, np.nan) / lot,
        )

    def _add_local_features(self, x: DataFrame) -> DataFrame:
        coords = np.deg2rad(x[["lat", "long"]].to_numpy())
        city_km = EARTH_RADIUS_KM * haversine_distances(coords, CITY_CENTERS)
        points = gpd.GeoSeries(gpd.points_from_xy(x["long"], x["lat"]), index=x.index, crs="EPSG:4326")
        lake_distance = points.to_crs(LAKE_CRS).distance(self.shoreline_).to_numpy()
        return self._house_features(x).assign(
            distance_to_seattle_km=city_km[:, 0],
            distance_to_bellevue_km=city_km[:, 1],
            lake_distance_m=lake_distance,
            view_log_lake_distance=x["view"] * np.log1p(lake_distance),
        )

    def _add_comparables(
        self, frame: DataFrame, neighborhoods: list[Positions], *, fold_ids: Positions | None = None
    ) -> DataFrame:
        medians = np.full(len(frame), np.nan)
        counts = np.zeros(len(frame), dtype=np.intp)
        ratios = self.reference_prices_["price_per_sqft_living"].to_numpy(dtype=float, na_value=np.nan)
        query_ids = frame["id"].to_numpy()
        query_attributes = frame[["sqft_living", "grade"]].to_numpy(dtype=float, na_value=np.nan)
        for i, neighbors in enumerate(neighborhoods):
            rows = neighbors
            if fold_ids is not None:
                rows = neighbors[fold_ids[neighbors] != fold_ids[i]]
            area, grade = query_attributes[i]
            ref_areas, ref_grades = self.comparable_attributes_[rows].T
            matches = (
                (self.comparable_ids_[rows] != query_ids[i])
                & (ref_areas > 0)
                & (np.abs(ref_areas - area) <= COMP_AREA_FRACTION * area)
                & (np.abs(ref_grades - grade) <= COMP_GRADE_GAP)
            )
            values = ratios[rows[matches]]
            values = values[np.isfinite(values)]
            counts[i] = values.size
            if values.size:
                medians[i] = np.median(values)
        return frame.assign(comp_median_price_per_sqft_living=medians, comp_count=counts)

    def _add_encoded_features(self, result: DataFrame) -> DataFrame:
        if self.encoder_ is not None:
            result[[f"{col}_mean" for col in self.columns_]] = self.encoder_.transform(result[self.columns_])
        return result

    def transform(self, x: DataFrame) -> DataFrame:
        """Append features from fitted references in query order."""
        check_is_fitted(self, "reference_prices_")
        result = self._add_encoded_features(self._add_local_features(x))
        return self._add_comparables(result, self.neighborhoods_.within_radius(x))

    def fit_transform(self, x: DataFrame, y: pd.Series) -> DataFrame:
        """Generate training features with whole houses held out by fold."""
        self.fit(x, y)
        result = self._add_local_features(x)

        categories = x[self.columns_]
        encoded = np.zeros(categories.shape)
        fold_ids = np.zeros(len(x), dtype=np.intp)
        n_groups = x["id"].nunique()
        if n_groups > 1:
            folds = GroupKFold(n_splits=min(self.n_splits, n_groups), shuffle=True, random_state=self.random_state)
            for fold, (train, valid) in enumerate(folds.split(x, groups=x["id"])):
                fold_ids[valid] = fold
                if self.encoder_ is not None:
                    encoder = clone(self.encoder_).fit(categories.iloc[train], y.iloc[train])
                    encoded[valid] = encoder.transform(categories.iloc[valid])
        if self.encoder_ is not None:
            result[[f"{col}_mean" for col in self.columns_]] = encoded
        return self._add_comparables(result, self.neighborhoods_.rows_, fold_ids=fold_ids)
