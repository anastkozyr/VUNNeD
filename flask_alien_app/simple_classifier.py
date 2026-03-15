import numpy as np
import hashlib
import json
import os


class SimpleAlienClassifier:
    """
    Простой классификатор на основе шаблонов и расстояния
    Не требует scikit-learn, tensorflow или других ML библиотек
    """

    def __init__(self):
        self.classes = ['Zeta Reticuli', 'Alpha Centauri', 'Betelgeuse', 'Andromeda',
                        'Sirius', 'Vega', 'Pleiades', 'Orion', 'Cygnus', 'Draco']
        self.templates = {}  # шаблоны для каждого класса
        self.is_trained = False

    def extract_features(self, signal):
        """
        Извлечение простых числовых характеристик из сигнала
        """
        # Базовые статистики
        features = [
            float(np.mean(signal)),
            float(np.std(signal)),
            float(np.max(signal)),
            float(np.min(signal)),
            float(np.median(signal)),
            float(np.percentile(signal, 25)),
            float(np.percentile(signal, 75)),
            float(np.sum(np.abs(signal))),  # энергия
            float(len(np.where(signal > np.mean(signal))[0])),  # точки выше среднего
        ]

        # Частотные характеристики (простая разность)
        if len(signal) > 1:
            diff = np.diff(signal)
            features.extend([
                float(np.mean(np.abs(diff))),
                float(np.std(diff)),
                float(np.max(np.abs(diff)))
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        return np.array(features)

    def create_template(self, signals):
        """
        Создание шаблона класса из нескольких сигналов
        """
        features_list = [self.extract_features(s) for s in signals]
        return np.mean(features_list, axis=0)

    def train(self, train_x, train_y):
        """
        Обучение модели - создание шаблонов для каждого класса
        """
        # Группируем сигналы по классам
        class_signals = {}
        for signal, label in zip(train_x, train_y):
            if label not in class_signals:
                class_signals[label] = []
            class_signals[label].append(signal)

        # Создаем шаблон для каждого класса
        for label, signals in class_signals.items():
            self.templates[int(label)] = self.create_template(signals)

        self.is_trained = True
        print(f"Модель обучена на {len(train_x)} сигналах, {len(self.templates)} классов")
        return True

    def predict(self, signals):
        """
        Предсказание класса для новых сигналов
        """
        predictions = []

        for signal in signals:
            # Извлекаем признаки
            features = self.extract_features(signal)

            if not self.is_trained:
                # Демо режим - случайный класс
                predictions.append(np.random.randint(0, len(self.classes)))
            else:
                # Находим ближайший шаблон по евклидову расстоянию
                min_dist = float('inf')
                best_class = 0

                for class_id, template in self.templates.items():
                    # Вычисляем евклидово расстояние
                    dist = np.linalg.norm(features - template)

                    # Нормализованное расстояние
                    if dist < min_dist:
                        min_dist = dist
                        best_class = class_id

                predictions.append(best_class)

        return np.array(predictions)

    def predict_with_confidence(self, signals):
        """
        Предсказание с уверенностью
        """
        predictions = []
        confidences = []

        for signal in signals:
            features = self.extract_features(signal)

            if not self.is_trained:
                # Демо режим
                pred = np.random.randint(0, len(self.classes))
                conf = np.random.random() * 0.3 + 0.6  # 0.6-0.9
                predictions.append(pred)
                confidences.append(conf)
            else:
                # Вычисляем расстояния до всех шаблонов
                distances = []
                for class_id, template in self.templates.items():
                    dist = np.linalg.norm(features - template)
                    distances.append((class_id, dist))

                # Сортируем по расстоянию
                distances.sort(key=lambda x: x[1])

                # Лучший класс
                best_class, best_dist = distances[0]

                # Вычисляем уверенность (чем меньше расстояние, тем выше уверенность)
                if len(distances) > 1:
                    second_dist = distances[1][1]
                    if best_dist > 0:
                        confidence = 1.0 - (best_dist / (best_dist + second_dist))
                    else:
                        confidence = 0.95
                else:
                    confidence = 0.8

                predictions.append(best_class)
                confidences.append(min(0.99, confidence))

        return np.array(predictions), np.array(confidences)


# Создаем глобальный экземпляр
classifier = SimpleAlienClassifier()


# Функция для создания демо-данных
def create_sample_data(n_samples=1000, signal_length=100):
    """
    Создание синтетических данных для демонстрации
    """
    np.random.seed(42)

    signals = []
    labels = []

    for i in range(n_samples):
        # Генерируем класс
        class_id = i % 10

        # Базовый сигнал (синусоида с разной частотой для каждого класса)
        t = np.linspace(0, 1, signal_length)
        freq = 5 + class_id * 1.5
        signal = np.sin(2 * np.pi * freq * t)

        # Добавляем шум
        noise = np.random.normal(0, 0.1, signal_length)
        signal += noise

        # Добавляем особенности класса
        if class_id % 3 == 0:
            signal += 0.2 * np.sin(4 * np.pi * freq * t)
        if class_id % 5 == 0:
            signal += 0.15 * np.cos(2 * np.pi * (freq / 2) * t)

        signals.append(signal)
        labels.append(class_id)

    return np.array(signals), np.array(labels)


# Предобучаем модель
print("Создание демо-классификатора...")
train_x, train_y = create_sample_data(800, 200)
classifier.train(train_x, train_y)
print("Классификатор готов!")