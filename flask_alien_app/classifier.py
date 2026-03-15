import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


class AlienSignalClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.classes = ['Zeta Reticuli', 'Alpha Centauri', 'Betelgeuse', 'Andromeda',
                        'Sirius', 'Vega', 'Pleiades', 'Orion', 'Cygnus', 'Draco']
        self.is_trained = False

    def extract_features(self, signals):
        """
        Извлечение признаков из звуковых сигналов
        signals: numpy array формы (n_samples, signal_length)
        """
        features = []
        for signal in signals:
            # Статистические признаки
            feat = [
                np.mean(signal),
                np.std(signal),
                np.max(signal),
                np.min(signal),
                np.median(signal),
                np.percentile(signal, 25),
                np.percentile(signal, 75),
                np.sum(np.abs(signal)),  # энергия сигнала
                np.mean(np.abs(np.diff(signal))),  # средняя частота изменений
                np.sum(signal > np.mean(signal)),  # количество значений выше среднего
            ]

            # Спектральные признаки (простое FFT)
            if len(signal) > 1:
                fft_vals = np.abs(np.fft.rfft(signal))
                if len(fft_vals) > 10:
                    feat.extend([
                        np.mean(fft_vals[:10]),
                        np.std(fft_vals[:10]),
                        np.max(fft_vals[:10])
                    ])
                else:
                    feat.extend([0, 0, 0])
            else:
                feat.extend([0, 0, 0])

            features.append(feat)

        return np.array(features)

    def train(self, train_x, train_y, valid_x=None, valid_y=None):
        """
        Обучение модели
        """
        # Извлекаем признаки
        X_train = self.extract_features(train_x)

        # Кодируем метки
        y_train = self.label_encoder.fit_transform(train_y)

        # Масштабируем признаки
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Создаем и обучаем модель (MLPClassifier как простая нейросеть)
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=42,
            verbose=True
        )

        # Обучаем
        history = self.model.fit(X_train_scaled, y_train)

        # Если есть валидационные данные
        if valid_x is not None and valid_y is not None:
            X_valid = self.extract_features(valid_x)
            X_valid_scaled = self.scaler.transform(X_valid)
            y_valid = self.label_encoder.transform(valid_y)

            valid_score = self.model.score(X_valid_scaled, y_valid)
            print(f"Validation accuracy: {valid_score:.4f}")

        self.is_trained = True
        return history

    def predict(self, signals):
        """
        Предсказание классов для новых сигналов
        """
        if not self.is_trained:
            # Демо режим - случайные предсказания
            return np.random.randint(0, len(self.classes), len(signals))

        X = self.extract_features(signals)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions

    def predict_proba(self, signals):
        """
        Предсказание вероятностей
        """
        if not self.is_trained:
            # Демо режим - случайные вероятности
            n_samples = len(signals)
            n_classes = len(self.classes)
            probs = np.random.rand(n_samples, n_classes)
            return probs / probs.sum(axis=1, keepdims=True)

        X = self.extract_features(signals)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save_model(self, filepath):
        """Сохранение модели"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'classes': self.classes,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath):
        """Загрузка модели"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.classes = model_data['classes']
            self.is_trained = model_data['is_trained']
            return True
        return False


# Создаем глобальный экземпляр классификатора
classifier = AlienSignalClassifier()


# Для демонстрации создадим синтетические данные обучения
def create_sample_training_data():
    """Создание демонстрационных данных для обучения"""
    np.random.seed(42)

    # Создаем синтетические сигналы для 10 классов
    n_samples_per_class = 120
    n_classes = 10
    signal_length = 1000

    train_x = []
    train_y = []

    for class_id in range(n_classes):
        for _ in range(n_samples_per_class):
            # Базовый сигнал
            t = np.linspace(0, 1, signal_length)
            # Разные частоты для разных классов
            freq = 5 + class_id * 2
            signal = np.sin(2 * np.pi * freq * t)

            # Добавляем шум и особенности
            noise = np.random.normal(0, 0.1, signal_length)
            signal += noise

            # Добавляем гармоники для разнообразия
            if class_id % 2 == 0:
                signal += 0.3 * np.sin(4 * np.pi * freq * t)

            train_x.append(signal)
            train_y.append(class_id)

    return np.array(train_x), np.array(train_y)


# Обучаем модель при импорте
print("Обучение классификатора...")
train_x, train_y = create_sample_training_data()
classifier.train(train_x, train_y)
print("Модель обучена!")