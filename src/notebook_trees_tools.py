"""Вспомогательные функции для ноутбука о решающих деревьях."""

from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
from pandas.io.formats.style import Styler
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

__all__ = [
    "animate_decision_surface",
    "calc_metrics",
    "get_meshgrid",
    "highlight_best",
    "plot_classification_dependence",
    "plot_decision_surface",
    "plot_dependence",
    "plot_regression_results",
    "visualize_metric_df",
]

_Metric = Callable[[np.ndarray, np.ndarray], float]
_ParameterGrid = Union[np.ndarray, Sequence[int]]


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
        numeric_series.max() if series.name in ["R2 (test)", "R2 (train)"] else numeric_series.min()
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
        [{"selector": "td, th", "props": [("border", "1px solid black")]}]
    ).apply(highlight_best, axis=0)


def plot_regression_results(
    X: np.ndarray,
    y: np.ndarray,
    X_grid: np.ndarray,
    f_grid: np.ndarray,
    model_preds: Optional[Dict[str, np.ndarray]] = None,
    title: str = "Сравнение предсказаний моделей",
    fig: Optional[plt.Figure] = None,
) -> None:
    """
    Отображает предсказания регрессионной модели вместе
    с истинной зависимостью и данными.

    Параметры:
    X (np.ndarray): Входные данные.
    y (np.ndarray): Целевые значения.
    X_grid (np.ndarray): Сетка для отображения исходной зависимости.
    f_grid (np.ndarray): Значения истинной зависимости на сетке X_grid.
    models (Optional[Dict[str, Optional[Any]]]): Словарь, где ключи — подписи, а
        значения — модели, реализующие метод predict (могут быть None).
    features_grid_list (Optional[List[np.ndarray]]): Список массивов сетки признаков
        для каждой модели.
    title (str): Заголовок графика (по умолчанию "Сравнение предсказаний моделей").
    fig (Optional[plt.Figure]): Фигура для отображения графика (по умолчанию None).
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(1, 1, 1)

    # Отображение данных
    sns.scatterplot(
        x=X.reshape(-1),
        y=y.reshape(-1),
        label="Данные",
        alpha=0.3,
        s=80,
        color="purple",
        ax=ax,
    )

    # Истинная зависимость
    sns.lineplot(
        x=X_grid,
        y=f_grid,
        label="Истинная зависимость",
        linewidth=5,
        alpha=0.4,
        ax=ax,
    )

    # Проверяем, переданы ли модели и соответствующие данные
    if model_preds:
        for label, y_pred in model_preds.items():
            mse = mean_squared_error(f_grid, y_pred.reshape(-1))
            sns.lineplot(
                x=X_grid,
                y=y_pred.reshape(-1),
                label=f"{label}, MSE: {mse:.4f}",
                linewidth=2,
                alpha=0.9,
                ax=ax,
            )

    ax.set_title(title)
    ax.set_xlabel("Признак $x$")
    ax.set_ylabel("Таргет")
    ax.legend(fontsize=10)


def get_meshgrid(
    data: np.ndarray, step: float, border: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Функция для получения сетки точек (x1, x2)
    для дальнейшего отображения их на графиках

    Параметры:
    data (np.ndarray) - входной датасет, набор точек (x1_i, x2_i).
    step (float) - шаг сетки.
    border (float) - отступ от минимальных и максимальных значений x1, x2 в data
    в сетке
    """
    x1_min, x1_max = data[:, 0].min() - border, data[:, 0].max() + border
    x2_min, x2_max = data[:, 1].min() - border, data[:, 1].max() + border
    return np.meshgrid(
        np.arange(x1_min, x1_max, step),
        np.arange(x2_min, x2_max, step),
    )


def plot_decision_surface(
    estimator: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fig: Optional[plt.Figure] = None,
    colors: str = "coolwarm",
    light_colors: str = "coolwarm",
    title: str = "",
    metrics: Sequence[_Metric] = (r2_score, mean_squared_error),
    step: float = 0.3,
    param1: str = "Признак 1",
    param2: str = "Признак 2",
) -> None:
    """
    Функция для отображения разделяющей поверхности

    Параметры:
    estimator (BaseEstimator) - модель.
    X_train, y_train (np.ndarray) - данные и разметка обучающей выборки.
    X_test, y_test (np.ndarray) - данные и разметка тестовой выборки.
    fig (Optional[plt.Figure]) - фигура для отображения графика.
    colors (str) - цветовая схема для отображения точек из разных классов.
    light_colors (str) - цветовая схема для отображения разделяющей поверхности.
    title - заголовок графика.
    metrics - метрики качества.
    step - шаг для get_meshgrid.
    param1 и param2 - названия параметров.
    """
    estimator.fit(X_train, y_train)  # обучаем модель
    x1_values, x2_values = get_meshgrid(X_train, step)
    x1_ravel, x2_ravel = x1_values.ravel(), x2_values.ravel()
    mesh_predictions_ravel = estimator.predict(np.c_[x1_ravel, x2_ravel])
    mesh_predictions = np.array(mesh_predictions_ravel).reshape(x1_values.shape)

    if fig is None:
        fig = plt.figure(figsize=(16, 6))

    # отображаем разделяющую поверхность и точки обучающей выборки
    ax = fig.add_subplot(1, 2, 1)
    ax.grid(False)
    ax.pcolormesh(x1_values, x2_values, mesh_predictions, cmap=light_colors)
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        s=50,
        alpha=0.7,
        cmap=colors,
        edgecolors="black",
    )
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    metric_title = "\n".join(
        "{}={:.2f}".format(metric.__name__, metric(y_train, estimator.predict(X_train)))
        for metric in metrics
    )
    ax.set_title(f"Обучающая выборка\n{metric_title}")

    # отображаем разделяющую поверхность и точки тестовой выборки
    ax = fig.add_subplot(1, 2, 2)
    ax.grid(False)
    ax.pcolormesh(x1_values, x2_values, mesh_predictions, cmap=light_colors)
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        s=50,
        alpha=0.7,
        cmap=colors,
        edgecolors="black",
    )
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    metric_title = "\n".join(
        "{}={:.2f}".format(metric.__name__, metric(y_test, estimator.predict(X_test)))
        for metric in metrics
    )
    ax.set_title(f"Тестовая выборка\n{metric_title}")
    fig.subplots_adjust(top=0.8)
    fig.suptitle(title, fontsize=18, y=0.98)


def _update_decision_surface(
    iter_num: int,
    *,
    fig: plt.Figure,
    estimator_class: Type[BaseEstimator],
    param_name: str,
    param_values: _ParameterGrid,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: Optional[int],
    metrics: Sequence[_Metric],
    step: float,
    param1: str,
    param2: str,
) -> None:
    """
    Функция для отрисовки очередного кадра анимации анимации.

    Параметры:
    iter_num (int) - номер итерации.
    """
    fig.clear()
    param_value = param_values[iter_num]
    estimator = estimator_class(random_state=random_state, **{param_name: param_value})
    plot_decision_surface(
        estimator,
        X_train,
        y_train,
        X_test,
        y_test,
        fig,
        title=f"{param_name}={param_value}",
        metrics=metrics,
        step=step,
        param1=param1,
        param2=param2,
    )


def animate_decision_surface(
    estimator_class: Type[BaseEstimator],
    param_name: str,
    param_values: _ParameterGrid,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    random_state: Optional[int] = 42,
    interval: int = 800,
    metrics: Sequence[_Metric] = (r2_score, mean_squared_error),
    step: float = 0.3,
    param1: str = "Признак 1",
    param2: str = "Признак 2",
) -> FuncAnimation:
    """Создаёт анимацию поверхности при изменении гиперпараметра дерева.

    Параметры:
    estimator_class: Класс решающего дерева для регрессии или классификации.
    param_name: Название изменяемого гиперпараметра.
    param_values: Последовательность значений гиперпараметра.
    X_train, y_train: Данные и разметка обучающей выборки.
    X_test, y_test: Данные и разметка тестовой выборки.
    random_state: Значение random_state для создаваемых моделей.
    interval: Интервал между кадрами в миллисекундах.
    metrics: Метрики, отображаемые в заголовках графиков.
    step: Шаг сетки для отображения поверхности.
    param1, param2: Подписи осей графиков.

    Возвращает:
    FuncAnimation: Анимация для отображения через HTML(anim.to_jshtml()).
    """
    fig = plt.figure(figsize=(14, 6))
    update = partial(
        _update_decision_surface,
        fig=fig,
        estimator_class=estimator_class,
        param_name=param_name,
        param_values=param_values,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=random_state,
        metrics=metrics,
        step=step,
        param1=param1,
        param2=param2,
    )
    anim = FuncAnimation(
        fig,
        update,
        frames=len(param_values),
        interval=interval,
        blit=False,
    )
    plt.close(fig)  # чтобы не дублировалось при отображении ниже
    return anim


def plot_dependence(
    param_name: str,
    grid: _ParameterGrid = range(2, 20),
    title: str = "",
    *,
    score: Callable[[str, _ParameterGrid], Tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Функция для отображения графика зависимости r2 и mse
    от значения параметра c названием param_name

    Параметры:
    param_name (str) - название параметра, который собираемся варьировать.
    grid (np.ndarray) - сетка значений параметра.
    title (str) - заголовок графика.
    score (Callable) - функция вычисляющая качество модели на сетке grid.
    """
    plt.figure(figsize=(14, 4))

    train_metrics, test_metrics = score(param_name, grid)

    plt.subplot(1, 2, 1)
    plt.plot(grid, train_metrics[:, 0], label="train", lw=2)
    plt.plot(grid, test_metrics[:, 0], label="test", lw=2)
    plt.xlabel(param_name)
    plt.ylabel("$R^2$ score")
    plt.legend(fontsize=12)

    plt.subplot(1, 2, 2)
    plt.plot(grid, train_metrics[:, 1], label="train mse", lw=2)
    plt.plot(grid, test_metrics[:, 1], label="test mse", lw=2)
    plt.legend(fontsize=12)
    plt.xlabel(param_name)
    plt.ylabel("MSE score")
    plt.suptitle(title, fontsize=16)
    plt.show()


def calc_metrics(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> List[float]:
    result = []
    for metric in [
        r2_score,
        mean_squared_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
    ]:
        result.append(metric(y_train, model.predict(X_train)))
        result.append(metric(y_test, model.predict(X_test)))
    return result


def plot_classification_dependence(
    param_name: str,
    grid: _ParameterGrid = range(2, 20),
    title: str = "",
    *,
    score: Callable[[str, _ParameterGrid], Tuple[List[float], List[float]]],
) -> None:
    """
    Функция для отображения графика зависимости accuracy
    от значения параметра c названием param_name

    Параметры:
    param_name (str) - название параметра, который собираемся варьировать.
    grid (np.ndarray) - сетка значений параметра.
    title (str) - заголовок графика.
    score (Callable) - функция вычисляющая качество модели на сетке grid.
    """
    plt.figure(figsize=(7, 4))

    train_metric, test_metric = score(param_name, grid)

    plt.plot(grid, train_metric, label="train", lw=2)
    plt.plot(grid, test_metric, label="test", lw=2)
    plt.xlabel(param_name)
    plt.ylabel("Accuracy score")
    plt.legend(fontsize=12)
    plt.title(title, fontsize=16)
    plt.show()
