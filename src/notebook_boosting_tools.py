"""Вспомогательные функции для ноутбука по градиентному бустингу."""

import os
import time
from typing import Any, Callable, List, Sequence, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from tqdm import trange


def measure_time(func: Callable[..., Any]) -> Callable[..., Tuple[float, Any]]:
    """
    Декоратор для измерения времени выполнения функции.
    Замеряет время, затраченное на выполнение обернутой функции, и возвращает
    кортеж (время_выполнения, результат).

    Параметры:
    - func (Callable): Функция, время выполнения которой нужно измерить.

     Возвращает:
      - Callable: Обернутая функция, возвращающая кортеж (время_выполнения, результат).
    """

    def wrapper(*args: Any, **kwargs: Any) -> Tuple[float, Any]:
        start_time = time.time()
        result = func(*args, **kwargs)
        return time.time() - start_time, result

    return wrapper


def save_model_and_get_size(model: Any, filename: str = "model.pkl") -> float:
    """
    Сохраняет модель в файл и возвращает её размер в килобайтах.

    Параметры:
    - model: Объект модели, которую нужно сохранить.
    - filename (str): Имя файла для сохранения. По умолчанию 'model.pkl'.

    Возвращает:
     - float: Размер файла модели в килобайтах.
    """
    joblib.dump(model, filename)
    return os.path.getsize(filename) / 1024


def evaluate_model_performance(
    model: BaseEstimator,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_iter: int = 10,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Функция для оценки модели в задаче регресии.

    Параметры:
    - model: необученная модель
    - X_train: тренировочные данные (признаки)
    - y_train: тренировочные данные (целевая переменная)
    - X_test: тестовые данные (признаки)
    - y_test: тестовые данные (целевая переменная)
    - n_iter: число интераций замеров параметров для оценки среднего

    Возвращает:
    - fit_times: время обучения
    - prediсt_times: время предсказания
    - sizes: память занимаемая обученной моделью
    - metrics: качество по метрике R2
    """
    fit_times = []
    predict_times = []
    sizes = []
    metrics = []

    for _ in range(n_iter):
        # Замер времени обучения
        fit_time, _ = measure_time(model.fit)(X_train, y_train)
        fit_times.append(fit_time)

        # Замер времени предсказания
        predict_time, y_pred = measure_time(model.predict)(X_test)
        predict_times.append(predict_time)

        # Сохранение модели и получение размера
        model_size_kb = save_model_and_get_size(model)
        sizes.append(model_size_kb)

        # Рассчет метрики
        metrics.append(r2_score(y_test, y_pred))

    return fit_times, predict_times, sizes, metrics


def plot_compare_model_performance(
    models: List[BaseEstimator],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    labels: List[str],
    colors: List[str],
) -> None:
    """
    Оценивает производительность моделей и строит графики сравнения.

    Параметры:
    - models : list
        Список моделей для оценки.
    - model_names : list
        Названия моделей для подписей на графике.
    - X_train, X_test, y_train, y_test : array-like
        Данные для обучения и тестирования.
    """
    # Оцениваем каждую модель
    results = []
    for model in models:
        res = evaluate_model_performance(model, X_train, X_test, y_train, y_test)
        results.append(res)

    # Настройки для графиков
    titles = [
        "Среднее время обучения",
        "Среднее время предсказания",
        "Размер модели",
        "Качество предсказания",
    ]
    ylabels = ["Время, с", "Время, с", "KB", "R2"]

    # Строим графики
    plt.figure(figsize=(40, 20))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        means = [np.mean(res[i]) for res in results]

        plt.bar(labels, means, color=colors)
        plt.ylabel(ylabels[i], fontsize=34)
        plt.xticks(weight="bold", fontsize=34)
        plt.title(titles[i], fontsize=40)


def plot_sample_predictions(
    sample: tuple[np.ndarray, np.ndarray],
    prediction: tuple[np.ndarray, np.ndarray],
    *,
    ax: Axes | None = None,
    labels: tuple[str | None, str | None] = ("Истина", "Предсказание"),
    ylabel: str = "Таргет",
    title: str = "",
) -> None:
    """Рисует выборку и заданную кривую с единым оформлением.

    Параметры:
    - sample: Пара (признаки, ответы) для точек выборки.
    - prediction: Пара (сетка, значения кривой) для линии предсказаний.
    - ax: Оси графика. По умолчанию используются текущие оси.
    - labels: Подписи выборки и кривой в легенде. None скрывает подпись.
    - ylabel: Подпись вертикальной оси. По умолчанию "Таргет".
    - title: Заголовок графика.

    Горизонтальная ось всегда подписана "Признак". Создание фигуры,
    размещение подграфиков и показ остаются в ноутбуке.
    """
    if ax is None:
        ax = plt.gca()

    X, y = sample
    grid, predictions = prediction
    data_label, prediction_label = labels

    ax.scatter(X, y, label=data_label, s=20, alpha=0.7)
    ax.plot(grid, predictions, color="red", linewidth=3, label=prediction_label)
    ax.set_xlabel("Признак", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(labelsize=12)
    if data_label is not None or prediction_label is not None:
        ax.legend(fontsize=12, loc="best")


def train_model_multiple_times(
    model: Any,
    x_generator: Callable[[int], np.ndarray],
    y: Callable[[np.ndarray], np.ndarray],
    n_iter: int = 500,
    sample_size: int = 500,
    example_point: np.ndarray = None,
) -> tuple[np.ndarray, float]:
    """
    Функция применяет модель n_iter раз к данным и возвращает предсказания.

    * model - модель
    * x_generator - функция генерирующая выборку
    * y - функция подсчета таргета
    * n_iter - количество применений
    * sample_size - размер выборки
    * example_point - точка для предсказания

    Returns:
    * Y_pred - массив предсказаний
    * Y_true - истинное значение
    """

    if example_point is None:
        example_point = [[1, 2, 3, 4, 5, 6]]

    Y_pred = []

    for i in trange(n_iter):
        x_train = x_generator(sample_size)
        y_train = y(x_train) + sps.norm(scale=1).rvs([sample_size])
        fitted = model.fit(x_train, y_train)
        Y_pred.append(fitted.predict(example_point)[0])

    Y_true = y(np.asarray(example_point))[0]

    return np.array(Y_pred), Y_true


def draw_predictions(
    model: Any,
    x_generator: Callable[[int], np.ndarray],
    y: Callable[[np.ndarray], np.ndarray],
    label: str,
    n_iter: int = 500,
    log: bool = True,
    show: bool = True,
    sample_size: int = 500,
    color: str = "blue",
    xlim: tuple[float, float] = (-100, 100),
    new_figure: bool = True,
    title: str = "",
) -> None:
    """
    Функция рисует kde предсказаний и выводит статистики.

    * Y_pred - массив предсказаний
    * Y_true - истинное значение
    * label - label кривой на графике
    * log - логирование статистик в stdout
    * show - будет ли вызван plt.show
    * color - цвет графика
    * xlim - пределы по x
    * new_figure - создавать новый график
    * title - заголовок графика
    * model - модель (для логирования)
    """
    Y_pred, Y_true = train_model_multiple_times(
        model=model,
        x_generator=x_generator,
        y=y,
        n_iter=n_iter,
        sample_size=sample_size,
    )

    if new_figure:
        plt.figure(figsize=(12, 5))

    # Рисуем KDE предсказаний
    sns.kdeplot(Y_pred, fill=True, color=color, label="kde " + label, alpha=0.35, linewidth=0)

    # Отмечаем среднее значение предсказаний
    plt.plot([np.mean(Y_pred)], [0], marker="^", ms=20, color=color)

    # Отмечаем истинное значение
    if show:
        plt.axvline([Y_true], color="red", label="$y(x)$")

    plt.title(title, fontsize=18)
    plt.xlabel("Область значений модели")
    plt.xlim(xlim)
    plt.legend(fontsize=12)

    if show:
        plt.legend(loc="best")
        plt.show()

    if log:
        print("model =", model if model else "Не указана")
        print("x_new = [1, 2, 3, 4, 5, 6]")
        print("Var(x_new.T @ theta) =", np.var(Y_pred))
        print("bias(x_new.T @ theta) =", np.mean(Y_pred) - Y_true)
        print("MSE =", np.mean((Y_pred - Y_true) ** 2))


def plot_compare_estimators(
    param_grid: Sequence[Union[float, int, str]],
    metrics: Sequence[Sequence[float]],
    metrics_label: Sequence[str],
    param_label: str = "",
) -> None:
    """Функция для построения графиков зависимости целевой метрики
    от некоторого параметра модели на обучающей и на валидационной
    выборке.

    :param estimator_labels: массив названий моделей
    :param param_grid: значения исследуемого параметра
    :param metrics: массив, где i-ый элемент -
        это массив значений i-ой характеристики модели
    :param param_label: названия параметра
    :param metrics_label: название метрики
    :param title: заголовок для графика
    """

    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i + 1)
        plt.plot(
            param_grid,
            metric,
            linewidth=3,
            alpha=0.9,
        )
        plt.xlabel(param_label)
        plt.ylabel(metrics_label[i])
        plt.tight_layout()
