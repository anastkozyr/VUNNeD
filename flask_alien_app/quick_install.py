#!/usr/bin/env python3
import subprocess
import sys
import os


def main():
    print("=" * 50)
    print("Установка Alien Signal Classifier")
    print("=" * 50)

    # Проверяем Python
    print(f"Python версия: {sys.version}")

    # Устанавливаем пакеты
    packages = [
        "Flask==2.3.3",
        "Flask-SQLAlchemy==3.1.1",
        "Flask-Login==0.6.2",
        "numpy==1.24.3",
        "pandas==2.0.3",
        "plotly==5.17.0",
        "scikit-learn==1.3.0",  # Правильное имя!
        "pytest==7.4.2",
        "python-dotenv==1.0.0",
        "email-validator==2.1.0",
        "bcrypt==4.0.1",
        "joblib==1.3.2",
        "scipy==1.11.3"
    ]

    for package in packages:
        print(f"Установка {package}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package])
        if result.returncode != 0:
            print(f"Ошибка при установке {package}")
            return False

    print("\n" + "=" * 50)
    print("Установка завершена успешно!")
    print("=" * 50)
    print("\nДля запуска приложения выполните:")
    print("python app.py")

    return True


if __name__ == "__main__":
    main()