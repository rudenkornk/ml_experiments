"""Вспомогательные функции для ноутбука по градиентному бустингу."""

import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import Pool
from IPython.display import display
from pandas.io.formats.style import Styler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import DMatrix


def train_model(
    model: Any,
    train_data: Union[Any, DMatrix, Pool],
    train_labels: Optional[Any] = None,
    num_boost_round: int = 20,
    obj: Optional[Any] = None,
    evals: Optional[list] = None,
    verbose_eval: bool = False,
    **kwargs,
) -> Tuple[Any, float]:
    """
    Универсальная функция для обучения моделей разных библиотек.

    Параметры:
    - model: экземпляр модели. Если None и train_data — DMatrix, используется XGBoost.
    - train_data: обучающая выборка (X, DMatrix или Pool)
    - train_labels: метки классов (если не переданы через DMatrix/Pool)
    - num_boost_round: количество бустинговых итераций для XGBoost
    - obj: функция ошибки для XGBoost (опционально)
    - evals: список пар (данные, название) для оценки на валидации в XGBoost
    - verbose_eval: флаг печати промежуточных результатов XGBoost
    - **kwargs: дополнительные параметры, пробрасываемые в `.fit()` для sklearn

    Возвращает:
    - model: обученная модель
    - train_time: время обучения в секундах
    """
    start_time = time.time()

    # Обработка XGBoost
    if isinstance(train_data, DMatrix):
        model = xgb.train(
            kwargs.get("params", {}),
            train_data,
            num_boost_round=num_boost_round,
            obj=obj,
            evals=evals,
            verbose_eval=verbose_eval,
        )

    # Обработка CatBoost с типом данных Pool
    elif isinstance(train_data, Pool):
        model.fit(train_data)

    # Обработка sklearn-подобных моделей
    else:
        fit_params = {}
        if "eval_set" in kwargs:
            fit_params["eval_set"] = kwargs["eval_set"]

        model.fit(train_data, train_labels, **fit_params)

    train_time = time.time() - start_time
    return model, train_time


def evaluate_model(
    model: Any,
    X_test: Union[Any, DMatrix, Pool],
    y_test: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float]:
    """
    Универсальная функция для оценки качества модели.

    Параметры:
    - model: обученная модель
    - X_test: тестовые данные (может быть обычный X, DMatrix или Pool)
    - y_test: истинные метки классов (если отсутствуют — извлекаются из DMatrix/Pool)

    Возвращает:
    - pred_time: время предсказания
    - accuracy: точность
    - precision: точность положительного класса
    - recall: полнота
    - f1: f1-мера
    """
    start_time = time.time()

    # Предсказание и извлечение y_test при необходимости

    # В случае когда X_test является типом данных DMatrix
    if isinstance(X_test, DMatrix):
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        if y_test is None:
            y_test = X_test.get_label()

    # В случае когда X_test является типом данных Pool
    elif isinstance(X_test, Pool):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        if y_test is None:
            y_test = X_test.get_label()

    # В остальных случаях
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        )

    pred_time = time.time() - start_time

    return (
        pred_time,
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred, zero_division=0),
    )


def highlight_best(series: pd.Series) -> List[str]:
    """
    Подсвечивает лучшие значения в столбце:
    - минимальные для времени;
    - максимальные для метрик.
    Игнорирует столбец 'Модель' и строковые 'nan'.

    Возвращает:
    - Список CSS-стилей для каждой ячейки.
    """
    if series.name == "Модель":
        return [""] * len(series)

    # Преобразование к числам, замена строковых 'nan'
    try:
        numeric_series = pd.to_numeric(series.replace(["nan", "NaN"], np.nan))
    except Exception:
        return [""] * len(series)

    if numeric_series.isnull().all():
        return [""] * len(series)

    # Выбор лучшего значения
    best_value = (
        numeric_series.min()
        if series.name in ["fit time, s", "pred time, s"]
        else numeric_series.max()
    )
    return ["background-color: lightgreen" if val == best_value else "" for val in numeric_series]


def visualize_metric_df(df: pd.DataFrame, precision: int = 4) -> Styler:
    """
    Визуализирует таблицу метрик с округлением, чёрными рамками и подсветкой лучших значений.

    Параметры:
    - df: DataFrame с метриками
    - precision: количество знаков после запятой

    Возвращает:
    - pandas Styler: стилизованная таблица
    """
    df_rounded = df.copy()
    numeric_cols = df_rounded.select_dtypes(include=["float64", "int64"]).columns

    df_rounded[numeric_cols] = df_rounded[numeric_cols].astype(float).round(precision)
    for col in numeric_cols:
        df_rounded[col] = df_rounded[col].apply(lambda x: f"{x:.{precision}f}")

    return df_rounded.style.set_table_styles(
        [
            {
                "selector": "td, th",
                "props": [("border", "1px solid black")],
            }
        ]
    ).apply(highlight_best, axis=0)


def evaluate_and_store(
    model_name: str,
    model: Any,
    X_test: Any,
    y_test: Any,
    train_time: float,
    results_df: pd.DataFrame,
    class_names: List[str] = ["Фон", "Сигнал"],
    precision: int = 4,
):
    """
    Выполняет оценку модели, сохраняет метрики в results_df, визуализирует таблицу и confusion matrix.

    Параметры:
    - model_name: имя модели
    - model: обученная модель
    - X_test, y_test: тестовые данные
    - train_time: время обучения
    - results_df: таблица с результатами (будет обновлена)
    - class_names: подписи классов для матрицы ошибок
    - precision: точность округления при визуализации таблицы
    """
    # Получаем все метрики в виде списка
    # функция возвращает список [pred_time, acc, prec, rec, f1, roc_auc, cm]
    metrics = evaluate_model(model, X_test, y_test)

    # Записываем результаты в DataFrame
    # Исключаем cm (матрицу ошибок) из записи в таблицу
    results_df.loc[len(results_df)] = [model_name, train_time] + list(metrics)

    # Отображаем результаты метрик
    display(visualize_metric_df(results_df, precision))
