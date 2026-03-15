import unittest
import os
import sys
import tempfile
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app, db, User
from simple_classifier import SimpleAlienClassifier, create_sample_data


class TestAlienApp(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app = app.test_client()
        with app.app_context():
            db.create_all()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def test_index_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_classifier_features(self):
        clf = SimpleAlienClassifier()
        signal = np.random.randn(100)
        features = clf.extract_features(signal)
        self.assertEqual(len(features), 12)  # 9 + 3 признака

    def test_classifier_prediction(self):
        clf = SimpleAlienClassifier()
        # Создаем тестовые данные
        train_x, train_y = create_sample_data(100, 50)
        clf.train(train_x, train_y)

        # Тестируем предсказание
        test_signal = train_x[0:1]
        pred = clf.predict(test_signal)
        self.assertTrue(0 <= pred[0] < 10)

    def test_classifier_confidence(self):
        clf = SimpleAlienClassifier()
        train_x, train_y = create_sample_data(100, 50)
        clf.train(train_x, train_y)

        test_signal = train_x[0:1]
        pred, conf = clf.predict_with_confidence(test_signal)
        self.assertTrue(0 <= conf[0] <= 1)


if __name__ == '__main__':
    unittest.main()