"""Вспомогательные функции и визуализации для ноутбука по логистической регрессии."""

import os
import time
from typing import Any, Callable, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.io.formats.style import Styler
from sklearn.base import BaseEstimator


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


def plot_compare_model_performance(
    models: List[BaseEstimator],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    labels: List[str],
    colors: List[str],
    *,
    evaluate_model_performance: Callable[..., Any],
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
    - evaluate_model_performance (Callable):
        Функция оценки производительности, определённая в ноутбуке.
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
    ylabels = ["Время, с", "Время, с", "KB", "Accuracy"]

    # Строим графики
    with sns.axes_style("white"):
        plt.figure(figsize=(9, 5))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            means = [np.mean(res[i]) for res in results]
            stds = [np.std(res[i]) for res in results]

            plt.bar(labels, means, yerr=stds, color=colors)
            plt.ylabel(ylabels[i], fontsize=10)
            plt.yticks(fontsize=10)
            plt.xticks(weight="bold", fontsize=10)
            plt.title(titles[i])
            if i < 3:
                plt.yscale('log')
            plt.tight_layout()

        plt.show()


class ModelBenchmark:
    """
    Класс для оценки, сравнения и визуализации метрик моделей машинного обучения.
    """

    def __init__(self, evaluate_model_performance: Callable[..., Any]) -> None:
        """
        Инициализирует пустой DataFrame с колонками для метрик.

        Параметры:
        - evaluate_model_performance (Callable): Функция оценки производительности,
            определённая в ноутбуке.
        """
        self._evaluate_model_performance = evaluate_model_performance
        self.results_df = pd.DataFrame(
            columns=["Model", "Fit time (s)", "Predict time (s)", "Memory (KB)", "Accuracy"]
        )

    def add_model(
        self, model: Any, model_name: str, X_train: Any, X_test: Any, y_train: Any, y_test: Any
    ) -> None:
        """
        Оценивает модель и добавляет результаты в таблицу.

        Args:
            model: Объект модели (sklearn-совместимый).
            model_name (str): Название модели для отчета.
            X_train, X_test, y_train, y_test: Данные для обучения и теста.
        """
        # Вызываем функцию оценки
        fit_times, predict_times, mem_sizes, metrics = self._evaluate_model_performance(
            model, X_train, X_test, y_train, y_test
        )

        # Создаем строку с результатами (берем среднее по фолдам/запускам)
        new_row = pd.DataFrame(
            {
                "Model": [model_name],
                "Fit time (s)": [np.mean(fit_times)],
                "Predict time (s)": [np.mean(predict_times)],
                "Memory (KB)": [np.mean(mem_sizes)],
                "Accuracy": [np.mean(metrics)],
            }
        )

        # Добавляем новую строку к общему датафрейму
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
        print(f"Модель '{model_name}' успешно добавлена.")

    def show_table(self, precision: int = 4) -> Styler:
        """
        Визуализирует таблицу метрик с округлением, рамками и подсветкой лучших значений.

        Args:
            precision (int): Количество знаков после запятой.

        Returns:
            pd.io.formats.style.Styler: Стилизованная таблица.
        """
        # Делаем копию, чтобы не менять основной dataframe
        df_display = self.results_df.copy()

        numeric_cols = df_display.select_dtypes(include=["float64", "int64"]).columns

        # Округляем и превращаем в строки для жесткого форматирования (как в оригинале)
        df_display[numeric_cols] = df_display[numeric_cols].astype(float).round(precision)

        for col in numeric_cols:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.{precision}f}")

        return df_display.style.set_table_styles(
            [{"selector": "td, th", "props": [("border", "1px solid black")]}]
        ).apply(self._highlight_best, axis=0)

    @staticmethod
    def _highlight_best(series: pd.Series) -> List[str]:
        """
        Внутренний метод для подсветки лучших значений в столбце.

        Логика:
        - Минимальные значения подсвечиваются для времени и памяти.
        - Максимальные значения подсвечиваются для метрик (Accuracy).

        Args:
            series (pd.Series): Столбец данных.

        Returns:
            List[str]: Список CSS-стилей для каждой ячейки столбца.
        """
        # Пропускаем столбец с именами моделей
        if series.name == "Model":
            return [""] * len(series)

        # Преобразование к числам, так как в visualize_metric_df они были конвертированы в строки
        try:
            numeric_series = pd.to_numeric(series.replace(["nan", "NaN"], np.nan))
        except Exception:
            return [""] * len(series)

        if numeric_series.isnull().all():
            return [""] * len(series)

        # Определяем критерий лучшего значения
        columns_to_minimize = ["Fit time (s)", "Predict time (s)", "Memory (KB)"]

        if series.name in columns_to_minimize:
            best_value = numeric_series.min()
        else:
            best_value = numeric_series.max()

        # Формируем список стилей
        # Используем np.isclose для сравнения float, чтобы избежать проблем с точностью
        return [
            (
                "background-color: lightgreen"
                if np.isclose(val, best_value)
                else "" if not np.isnan(val) else ""
            )
            for val in numeric_series
        ]


def visualize_data(X: np.ndarray, y: Union[np.ndarray, list]) -> None:
    """Визуализирует данные на основе числа признаков в матрице X.

    Если X имеет один признак (n_features = 1), отображается гистограмма для каждого класса.
    Если X имеет два признака (n_features = 2), отображается scatter plot для каждого класса.

    Параметры:
    X (np.ndarray): Двумерный массив признаков данных. Размерность (n_samples, n_features).
    y (Union[np.ndarray, list]): Вектор меток классов данных. Размерность (n_samples,).
    """

    # Определяем число признаков
    n_features = X.shape[1]

    if n_features == 1:
        # Гистограмма для данных с одним признаком
        plt.hist(X[y == 0], bins=12, alpha=0.6, label="Класс 0")
        plt.hist(X[y == 1], bins=12, alpha=0.6, label="Класс 1")
        plt.xlabel("Значение признака")
        plt.ylabel("Количество элементов")
        plt.title("Распределение синтетических данных по классам")

    elif n_features == 2:
        # Точечная диаграмма для данных с двумя признаками
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], s=70, alpha=0.5, label="Класс 0")
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=70, alpha=0.5, label="Класс 1")
        plt.xlabel("Признак 1")
        plt.ylabel("Признак 2")
        plt.title("Синтетические данные для бинарной классификации")

    else:
        raise ValueError("Функция поддерживает только 1 или 2 признака для визуализации.")

    plt.legend(fontsize=15)
    plt.show()


def plot_logreg_coefficients(coef_plot_df: pd.DataFrame) -> None:
    """
    Строит графики коэффициентов логистической регрессии и отношений шансов.

    Параметры:
    coef_plot_df (pd.DataFrame): Подготовленная таблица со столбцами «Признак»,
        «Коэффициент» и «Отношение_шансов» в порядке отображения на графиках.
    """
    with sns.axes_style("white"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

    # Верхний график — коэффициенты
    colors = ["green" if x > 0 else "red" for x in coef_plot_df["Коэффициент"]]
    ax1.barh(coef_plot_df["Признак"], coef_plot_df["Коэффициент"], color=colors, alpha=0.7)
    ax1.set_xlabel("Значение коэффициента")
    ax1.set_title("Коэффициенты логистической регрессии", fontsize=18)
    ax1.axvline(x=0, color="black", linestyle="-", alpha=0.3)
    ax1.tick_params(axis="y")
    ax1.tick_params(axis="x")

    for coef_index, coef in enumerate(coef_plot_df["Коэффициент"]):
        ax1.text(
            coef,
            coef_index,
            f"{coef:.2f}",
            va="center",
            ha="right" if coef > 0 else "left",
            fontweight="bold",
        )

    # Нижний график — отношения шансов
    ax2.barh(coef_plot_df["Признак"], coef_plot_df["Отношение_шансов"], color=colors, alpha=0.7)
    ax2.set_xlabel("Отношение шансов (exp(коэффициент))")
    ax2.set_title("Влияние признаков на вероятность дохода >50K", fontsize=18)
    ax2.axvline(x=1, color="black", linestyle="--", alpha=0.3, label="Нет влияния")
    ax2.tick_params(axis="y")
    ax2.tick_params(axis="x")

    for ind, ratio in enumerate(coef_plot_df["Отношение_шансов"]):
        if ratio > 1:
            interpretation = f"↑ шанс в {ratio:.2f} раза"
        else:
            interpretation = f"↓ шанс в {1/ratio:.2f} раза"
        ax2.text(ratio, ind, interpretation, va="center", ha="left")

    ax2.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_logit_linearity_check(
    X: np.ndarray,
    y: Union[np.ndarray, list],
    h: float = 1.0,
    size: Optional[int] = 5000,
    use_percentiles: bool = True,
    figsize: Tuple[int, int] = (12, 4),
    *,
    kernel_smooth_logit: Callable[..., Tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Визуальная проверка линейности логита для всех признаков.

    Параметры:
    X : np.ndarray, shape (n_samples, n_features)
        Матрица признаков данных.
    y : Union[np.ndarray, list], shape (n_samples,)
        Вектор меток классов (0 и 1).
    h : float, default=1.0
        Ширина ядра для сглаживания.
    size : Optional[int], default=5000
        Размер подвыборки для ускорения вычислений. Если None, используется вся выборка.
    use_percentiles : bool, default=True
        Если True, использует 5-й и 95-й процентили для построения сетки (устойчиво к выбросам).
        Если False, использует минимум и максимум.
    figsize : Tuple[int, int], default=(12, 4)
        Размер фигуры для отображения графиков.
    kernel_smooth_logit : Callable
        Функция вычисления сглаженного логита, определённая в ноутбуке.
    """
    y = np.asarray(y)

    n_features = X.shape[1]

    if size is None or size > len(X):
        size = len(X)

    plt.figure(figsize=figsize)

    for feature_idx in range(n_features):
        # Выбираем подвыборку для ускорения
        x_subset = X[:size, feature_idx]
        y_subset = y[:size]

        # Сетка точек
        if use_percentiles:
            x_min = np.percentile(x_subset, 5)
            x_max = np.percentile(x_subset, 95)
        else:
            x_min = x_subset.min()
            x_max = x_subset.max()

        x_grid = np.linspace(x_min, x_max, 100)

        # Вычисляем сглаженный логит
        _, logit_smoothed = kernel_smooth_logit(x_subset, y_subset, x_grid, h)

        # Строим график
        plt.subplot(1, n_features, feature_idx + 1)
        plt.plot(x_grid, logit_smoothed, lw=3)
        plt.xlabel(f"Признак {feature_idx}")
        plt.ylabel("Приближение логита", fontsize=16)

    plt.suptitle("Проверка линейности логита")
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Предсказания модели линейной регрессии",
    xlabel: str = "Признак 1",
    ylabel: str = "Признак 2",
    cmap: str = "summer",
    colorbar_label: str = "Вероятность класса 1",
) -> None:
    """Визуализирует предсказание модели бинарной классификации.

    Параметры:
    model (Any): Обученная модель с методом predict_proba.
    X (np.ndarray): Матрица признаков (только первые два признака используются для визуализации).
    y (np.ndarray): Целевые метки.
    figsize (Tuple[int, int]): Размер графика (по умолчанию (10, 6)).
    title (str): Заголовок графика (по умолчанию "Предсказания модели линейной регрессии").
    xlabel (str): Подпись оси X (по умолчанию "Признак 1").
    ylabel (str): Подпись оси Y (по умолчанию "Признак 2").
    cmap (str): Цветовая схема (по умолчанию "summer").
    colorbar_label (str): Подпись цветовой шкалы (по умолчанию "Вероятность класса 1").
    """

    # Установка границ
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Создание сетки
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))

    # Предсказание вероятностей для сетки
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Построение графика
    plt.figure(figsize=figsize)
    plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower", cmap=cmap, alpha=0.8)

    # Визуализация исходных точек
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=cmap, s=60, alpha=0.8)

    # Настройка оформления
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(False)
    plt.colorbar(label=colorbar_label)
    plt.show()
