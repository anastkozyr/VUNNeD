import numpy as np
import plotly
import plotly.graph_objs as go
import plotly.express as px
import json
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os


def load_model(model_path):
    """Загрузка предобученной модели"""
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            return model
        else:
            # Возвращаем заглушку если модели нет
            return None
    except:
        return None


def predict_signal(model, signals):
    """Предсказание класса сигнала"""
    if model is None:
        # Демо режим - случайные предсказания
        return np.random.randint(0, 10, len(signals))

    predictions = model.predict(signals)
    return np.argmax(predictions, axis=1)


def load_test_data(filepath):
    """Загрузка тестовых данных из .npz"""
    data = np.load(filepath)
    return data['test_x'], data['test_y']


def create_sample_charts():
    """Создание демонстрационных графиков для аналитики"""

    # 1. График точности обучения
    epochs = list(range(1, 21))
    train_acc = [0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.89, 0.91, 0.92, 0.93,
                 0.94, 0.95, 0.95, 0.96, 0.96, 0.97, 0.97, 0.97, 0.98, 0.98]
    val_acc = [0.60, 0.68, 0.74, 0.78, 0.81, 0.83, 0.85, 0.86, 0.87, 0.88,
               0.89, 0.89, 0.90, 0.90, 0.91, 0.91, 0.91, 0.92, 0.92, 0.92]

    accuracy_chart = {
        'data': [
            {'x': epochs, 'y': train_acc, 'type': 'scatter', 'name': 'Train Accuracy'},
            {'x': epochs, 'y': val_acc, 'type': 'scatter', 'name': 'Validation Accuracy'}
        ],
        'layout': {
            'title': 'Точность обучения по эпохам',
            'xaxis': {'title': 'Эпоха'},
            'yaxis': {'title': 'Точность'}
        }
    }

    # 2. Распределение классов в обучающем наборе
    classes = ['Zeta Reticuli', 'Alpha Centauri', 'Betelgeuse', 'Andromeda',
               'Sirius', 'Vega', 'Pleiades', 'Orion', 'Cygnus', 'Draco']
    class_counts = [145, 132, 118, 124, 108, 96, 112, 105, 87, 93]

    distribution_chart = {
        'data': [
            {'x': classes, 'y': class_counts, 'type': 'bar', 'name': 'Количество записей'}
        ],
        'layout': {
            'title': 'Распределение классов в обучающем наборе',
            'xaxis': {'title': 'Цивилизация'},
            'yaxis': {'title': 'Количество записей'}
        }
    }

    # 3. Точность на тестовом наборе
    test_classes = classes[:5]
    test_accuracy = [0.95, 0.92, 0.88, 0.91, 0.89]

    test_accuracy_chart = {
        'data': [
            {'x': test_classes, 'y': test_accuracy, 'type': 'bar', 'name': 'Точность'}
        ],
        'layout': {
            'title': 'Точность определения по классам',
            'xaxis': {'title': 'Цивилизация'},
            'yaxis': {'title': 'Точность', 'range': [0, 1]}
        }
    }

    # 4. Топ-5 классов
    top5_counts = [145, 132, 124, 118, 112]
    top5_classes = ['Zeta Reticuli', 'Alpha Centauri', 'Andromeda', 'Betelgeuse', 'Pleiades']

    top5_chart = {
        'data': [
            {'x': top5_classes, 'y': top5_counts, 'type': 'bar', 'name': 'Количество'}
        ],
        'layout': {
            'title': 'Топ-5 самых частых классов',
            'xaxis': {'title': 'Цивилизация'},
            'yaxis': {'title': 'Количество записей'}
        }
    }

    return {
        'accuracy_chart': accuracy_chart,
        'distribution_chart': distribution_chart,
        'test_accuracy_chart': test_accuracy_chart,
        'top5_chart': top5_chart
    }