import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import re


def restore_labels(labels):
    restored = []
    invalid = []
    for i, label in enumerate(labels):
        label_str = str(label)
        numbers = re.findall(r'\d+', label_str)
        if numbers:
            num = int(numbers[0])
            if num < 1000:
                restored.append(num)
            else:
                invalid.append((i, num, label_str))
                restored.append(-1)
        else:
            invalid.append((i, -1, label_str))
            restored.append(-1)

    if invalid:
        print(f"Найдено {len(invalid)} некорректных меток:")
        for i, num, text in invalid[:10]:
            print(f"  {i}: {text} -> {num}")

    return np.array(restored)


train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')

valid_x = np.load('valid_x.npy')
valid_y = np.load('valid_y.npy')

train_y = restore_labels(train_y)
valid_y = restore_labels(valid_y)

mask_train = train_y != -1
mask_valid = valid_y != -1

X_train = train_x[mask_train]
y_train = train_y[mask_train]

X_valid = valid_x[mask_valid]
y_valid = valid_y[mask_valid]

all_classes = np.unique(np.concatenate([y_train, y_valid]))
class_mapping = {old: new for new, old in enumerate(all_classes)}

y_train = np.array([class_mapping[val] for val in y_train])
y_valid = np.array([class_mapping[val] for val in y_valid])

num_classes = len(all_classes)
print(f'Количество классов: {num_classes}')
print(f'Форма обучающих данных: {X_train.shape}')
print(f'Форма валидационных данных: {X_valid.shape}')

model = models.Sequential([
    layers.Input(shape=(80000, 1)),
    layers.Conv1D(32, 5, activation='relu', padding='same'),
    layers.MaxPooling1D(4),
    layers.BatchNormalization(),
    layers.Conv1D(64, 5, activation='relu', padding='same'),
    layers.MaxPooling1D(4),
    layers.BatchNormalization(),
    layers.Conv1D(128, 5, activation='relu', padding='same'),
    layers.MaxPooling1D(4),
    layers.BatchNormalization(),
    layers.Conv1D(256, 5, activation='relu', padding='same'),
    layers.GlobalAveragePooling1D(),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

model.save('alien_classifier.h5')
print("Модель сохранена как alien_classifier.h5")

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
valid_loss, valid_acc = model.evaluate(X_valid, y_valid, verbose=0)
print(f'Точность на обучении: {train_acc:.4f}')
print(f'Точность на валидации: {valid_acc:.4f}')