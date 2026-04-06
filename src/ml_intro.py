# ruff: noqa: B018, D100, D101, D102, D103, E402, PLR2004

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ones_matrix = np.ones((5, 5))
ones_submatrix_view = ones_matrix[::2, ::2]
ones_matrix[::2, ::2] = np.zeros((3, 3))
ones_submatrix_view


def construct_matrix(first_array: np.ndarray, second_array: np.ndarray) -> np.ndarray:
    return np.column_stack((first_array, second_array))


construct_matrix(np.array([1, 2]), np.array([3, 4]))


construct_matrix(np.array([1, 2]), np.array([3, 4]))


p = np.arange(1).reshape([1, 1, 1, 1])
p


np.concatenate((p, p), axis=3).shape


def most_frequent(nums: np.ndarray) -> np.ndarray:
    counts = np.bincount(nums)
    return np.argmax(counts)


src = Path.cwd() / "src"
data_dir = Path.cwd() / "data" / "ml_intro"


data = pd.read_csv(str(data_dir / "organisations.csv"))
features = pd.read_csv(data_dir / "features.csv")
rubrics = pd.read_csv(data_dir / "rubrics.csv")


pd.set_option("display.max_columns", None)
data
features
rubrics
data.info()


feature_map = features.set_index("feature_id").to_dict()["feature_name"]


rubrics


b = data[data["average_bill"] < 10000]
b

plt.hist(b["average_bill"], bins=50)
plt.show()


data = data[~data["average_bill"].isna()]
data_filt = data[data["average_bill"] <= 2500]
data_filt


cafe = rubrics.set_index("rubric_name").to_dict()["rubric_id"]["Кафе"]
msk = data_filt[(data_filt["city"] == "msk") & (data_filt["rubrics_id"].str.contains(str(cafe)))]
spb = data_filt[(data_filt["city"] == "spb") & (data_filt["rubrics_id"].str.contains(str(cafe)))]
msk["average_bill"].mean() - spb["average_bill"].mean()


data_train, data_test = train_test_split(data_filt, stratify=data_filt["average_bill"], test_size=0.33, random_state=42)


import numpy as np
from sklearn.base import RegressorMixin


class MeanRegressor(RegressorMixin):
    def fit(self, _x: np.ndarray | None, y: np.ndarray) -> None:
        self._mean = np.mean(y)

    def predict(self, _x: np.ndarray) -> None:
        return np.full(shape=_x.shape[0], fill_value=self._mean)


import numpy as np
from sklearn.base import ClassifierMixin


class MostFrequentClassifier(ClassifierMixin):
    def fit(self, _: np.ndarray, y: np.ndarray) -> None:
        counts = np.bincount(y)
        self._mode = np.argmax(counts)

    def predict(self, _x: np.ndarray | None = None) -> None:
        return np.full(shape=_x.shape[0], fill_value=self._mode)


mean_model = MeanRegressor()
mean_model.fit(data_train.drop("average_bill", axis=1), data_train["average_bill"])


most_frequent_model = MostFrequentClassifier()
most_frequent_model.fit(data_train.drop("average_bill", axis=1), data_train["average_bill"])

from sklearn.metrics import balanced_accuracy_score, mean_squared_error

y_true = data_test["average_bill"]
y_mean_predicted = mean_model.predict(data_test.drop("average_bill", axis=1))
mean_squared_error(y_true, y_mean_predicted)


y_most_freq_predicted = most_frequent_model.predict(data_test.drop("average_bill", axis=1))
mean_squared_error(y_true, y_most_freq_predicted)
balanced_accuracy_score(y_true, y_most_freq_predicted)


import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin


class CityMeanRegressor(RegressorMixin):
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._city_means = pd.DataFrame(np.column_stack((x, y))).groupby(1)[5].mean().to_dict()

    def predict(self, x: np.ndarray) -> pd.Series:
        return x["city"].map(self._city_means).to_numpy()


city_model = CityMeanRegressor()
city_model.fit(data_train.drop("average_bill", axis=1), data_train["average_bill"])
y_city_predicted = city_model.predict(data_test.drop("average_bill", axis=1))
mean_squared_error(y_true, y_city_predicted)


rubrics_cnt = (data_train.groupby("rubrics_id").size() >= 100).to_dict()
rubrics_cnt = {el for el, val in rubrics_cnt.items() if val}
data_train["modified_rubrics"] = data_train["rubrics_id"].transform(lambda x: x if x in rubrics_cnt else "other")
data_test["modified_rubrics"] = data_test["rubrics_id"].transform(lambda x: x if x in rubrics_cnt else "other")


import numpy as np
import pandas as pd


class CityRubricMedianClassifier(ClassifierMixin):
    def fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        x["average_bill"] = y
        self._stats = x.groupby(["city", "modified_rubrics"])["average_bill"].median().to_dict()

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return x.apply(
            lambda row: self._stats[(row["city"], row["modified_rubrics"])],
            axis=1,
        ).to_numpy()


city_rubric_model = CityRubricMedianClassifier()
city_rubric_model.fit(data_train.drop("average_bill", axis=1), data_train["average_bill"])
y_city_rubric_predicted = city_rubric_model.predict(data_test.drop("average_bill", axis=1))
mean_squared_error(y_true, y_city_rubric_predicted)
balanced_accuracy_score(y_true, y_city_rubric_predicted)


data_train["modified_features"] = data_train.apply(lambda row: row["rubrics_id"] + " q " + row["features_id"], axis=1)
mod_features = set(data_train["modified_features"])

data_test["modified_features"] = data_test.apply(lambda row: row["rubrics_id"] + " q " + row["features_id"], axis=1)
data_test["modified_features"] = data_test.apply(
    lambda row: row["modified_features"] if row["modified_features"] in mod_features else "other", axis=1
)


class BadClassifier(ClassifierMixin):
    def fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        x["average_bill"] = y
        self._stats = x.groupby("modified_features")["average_bill"].median().to_dict()
        self._stats["other"] = x["average_bill"].median()

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return x.apply(
            lambda row: self._stats[row["modified_features"]],
            axis=1,
        ).to_numpy()


bad_model = BadClassifier()
bad_model.fit(data_train.drop("average_bill", axis=1), data_train["average_bill"])
bad_predicted = bad_model.predict(data_test.drop("average_bill", axis=1))

y_true_actual = data_train["average_bill"]
bad_actual = bad_model.predict(data_train.drop("average_bill", axis=1))


np.sqrt(mean_squared_error(y_true, bad_predicted))
balanced_accuracy_score(y_true, bad_predicted)
np.sqrt(mean_squared_error(y_true_actual, bad_actual))
balanced_accuracy_score(y_true_actual, bad_actual)

data_train["rubrics_lst"] = data_train["rubrics_id"].str.split().apply(lambda x: [int(el) for el in x])
data_train["features_lst"] = data_train["features_id"].str.split().apply(lambda x: [int(el) for el in x])
data_test["rubrics_lst"] = data_test["rubrics_id"].str.split().apply(lambda x: [int(el) for el in x])
data_test["features_lst"] = data_test["features_id"].str.split().apply(lambda x: [int(el) for el in x])

all_rubrics = set(data_train["rubrics_lst"].explode()) | set(data_test["rubrics_lst"].explode())
all_features = set(data_train["features_lst"].explode()) | set(data_test["features_lst"].explode())


sparse_data_train = pd.DataFrame()
sparse_data_train["city"] = data_train["city"].map(lambda x: int(x == "msk"))
sparse_data_train["rating"] = data_train["rating"]
for rubric in all_rubrics:
    cond = data_train["rubrics_lst"].apply(lambda x, rubric=rubric: rubric in x)
    sparse_data_train[rubric] = pd.Series(cond).astype(pd.SparseDtype(int, fill_value=0))

for feature in all_features:
    cond = data_train["features_lst"].apply(lambda x, feature=feature: feature in x)
    sparse_data_train[feature] = pd.Series(cond).astype(pd.SparseDtype(int, fill_value=0))
y_train = data_train["average_bill"].to_numpy()


sparse_data_test = pd.DataFrame()
sparse_data_test["city"] = data_test["city"].map(lambda x: int(x == "msk"))
sparse_data_test["rating"] = data_test["rating"]
for rubric in all_rubrics:
    cond = data_test["rubrics_lst"].apply(lambda x, rubric=rubric: rubric in x)
    sparse_data_test[rubric] = pd.Series(cond).astype(pd.SparseDtype(int, fill_value=0))

for feature in all_features:
    cond = data_test["features_lst"].apply(lambda x, feature=feature: feature in x)
    sparse_data_test[feature] = pd.Series(cond).astype(pd.SparseDtype(int, fill_value=0))
y_test = data_test["average_bill"].to_numpy()


from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(sparse_data_train, y_train_encoded)
y_pred_train_encoded = model.predict(sparse_data_train)
y_pred_train = encoder.inverse_transform(y_pred_train_encoded)
y_pred_test_encoded = model.predict(sparse_data_test)
y_pred_test = encoder.inverse_transform(y_pred_test_encoded)

np.sqrt(mean_squared_error(y_train, y_pred_train))
balanced_accuracy_score(y_train, y_pred_train)
np.sqrt(mean_squared_error(y_test, y_pred_test))
balanced_accuracy_score(y_test, y_pred_test)
