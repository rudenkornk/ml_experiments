"""Reusable geographic neighborhoods of Seattle training sales."""

from typing import Self

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0
type Positions = NDArray[np.intp]
type Floats = NDArray[np.float64]


class HouseNeighborhoods:
    """Index training sales by geographic distance."""

    def __init__(self, *, radius_km: float) -> None:
        """Set the geographic search radius."""
        self.radius_km = radius_km

    @staticmethod
    def _locations(frame: DataFrame) -> Floats:
        return np.deg2rad(frame[["lat", "long"]].to_numpy(dtype=float))

    def fit(self, reference: DataFrame) -> Self:
        """Cache matching sales for every reference sale."""
        if not 0 < self.radius_km < np.inf:
            msg = "Expected a positive, finite search radius."
            raise ValueError(msg)
        coords = self._locations(reference)
        self.tree_ = BallTree(coords, metric="haversine")
        self.rows_ = self._within_radius(coords)
        return self

    def _within_radius(self, coords: Floats) -> list[Positions]:
        if not len(coords):
            return []
        candidates = self.tree_.query_radius(coords, r=self.radius_km / EARTH_RADIUS_KM)
        return [np.sort(rows) for rows in candidates]

    def nearest(self, query: DataFrame) -> Positions:
        """Find the geographically closest sale for each query."""
        return self.nearest_k(query, k=1).ravel()

    def nearest_k(self, query: DataFrame, *, k: int) -> Positions:
        """Return the k closest sales for each query, ordered by geographic distance."""
        coords = self._locations(query)
        if not len(coords):
            return np.empty((0, k), dtype=np.intp)
        return self.tree_.query(coords, k=k, return_distance=False)

    def within_radius(self, query: DataFrame) -> list[Positions]:
        """Return only sales within the search radius for each query."""
        return self._within_radius(self._locations(query))

    def transform(self, query: DataFrame) -> list[Positions]:
        """Return all matching sale positions, or the closest sale when none match."""
        coords = self._locations(query)
        neighborhoods = self._within_radius(coords)
        missing = [i for i, rows in enumerate(neighborhoods) if not len(rows)]
        if missing:
            closest = self.tree_.query(coords[missing], k=1, return_distance=False)
            for i, rows in zip(missing, closest, strict=True):
                neighborhoods[i] = rows
        return neighborhoods
