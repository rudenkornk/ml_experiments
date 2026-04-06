# ruff: noqa: D100, D103

from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

data_dir = Path.cwd() / "data" / "runes"


known_raw = pd.read_csv(str(data_dir / "train_runes.csv"))
unknown_raw = pd.read_csv(data_dir / "test_runes.csv")

n_runes = 5


def transform_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame()
    for i in range(n_runes):
        res[i] = data["rune"].str[i].astype("category")

    if "spell" in data:
        res["spell"] = data["spell"].astype("category")

    return res


known = transform_raw_data(known_raw)
unknown = transform_raw_data(unknown_raw)

train, test = train_test_split(known, stratify=known["spell"], test_size=0.2, random_state=0)

model = XGBClassifier(enable_categorical=True)
model.fit(train.drop(columns="spell"), train["spell"])
y_pred_train = model.predict(train.drop(columns="spell"))
y_pred_test = model.predict(test.drop(columns="spell"))

balanced_accuracy_score(train["spell"], y_pred_train)
balanced_accuracy_score(test["spell"], y_pred_test)

unknown_pred = model.predict(unknown)
res = unknown_raw.copy()
res["spell"] = unknown_pred
