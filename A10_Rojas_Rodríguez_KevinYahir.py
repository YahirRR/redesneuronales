import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Cargar y preprocesar Fashion‑MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Añadir canal (grises) -> (28, 28, 1)
x_train = x_train[..., None]
x_test  = x_test[..., None]

num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test, num_classes)

# 2. Función para construir LeNet‑5
def build_lenet5(pooling="avg", activation="tanh",
                 input_shape=(28, 28, 1), num_classes=10):
    if pooling == "avg":
        Pool = layers.AveragePooling2D
    elif pooling == "max":
        Pool = layers.MaxPooling2D
    else:
        raise ValueError("pooling debe ser 'avg' o 'max'")

    model = models.Sequential()
    # Conv1: 6 filtros 5x5
    model.add(layers.Conv2D(6, kernel_size=(5, 5),
                            activation=activation,
                            padding="valid",
                            input_shape=input_shape))
    model.add(Pool(pool_size=(2, 2), strides=2))
    # Conv2: 16 filtros 5x5
    model.add(layers.Conv2D(16, kernel_size=(5, 5),
                            activation=activation,
                            padding="valid"))
    model.add(Pool(pool_size=(2, 2), strides=2))
    # Capas totalmente conectadas
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation=activation))
    model.add(layers.Dense(84, activation=activation))
    # Capa de salida con Softmax
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model

# 3. Construir las dos versiones

# Versión 1: LeNet‑5 clásica: AveragePooling + tanh + Softmax
model_avg_tanh = build_lenet5(pooling="avg", activation="tanh")

# Versión 2: LeNet‑5 modificada: MaxPooling + ReLU + Softmax
model_max_relu = build_lenet5(pooling="max", activation="relu")

# 4. Compilar modelos (misma configuración de entrenamiento)
for model in [model_avg_tanh, model_max_relu]:
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

# 5. Entrenar ambas versiones
epochs = 15
batch_size = 128

history_avg_tanh = model_avg_tanh.fit(
    x_train, y_train_cat,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    verbose=1
)

history_max_relu = model_max_relu.fit(
    x_train, y_train_cat,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    verbose=1
)

# 6. Evaluar en el conjunto de prueba
test_loss_avg, test_acc_avg = model_avg_tanh.evaluate(x_test, y_test_cat, verbose=0)
test_loss_max, test_acc_max = model_max_relu.evaluate(x_test, y_test_cat, verbose=0)

print(f"Versión AvgPool + tanh -> Test accuracy: {test_acc_avg:.4f}, Test loss: {test_loss_avg:.4f}")
print(f"Versión MaxPool + ReLU -> Test accuracy: {test_acc_max:.4f}, Test loss: {test_loss_max:.4f}")

# 7. Graficar Accuracy y Loss vs Epochs de ambas versiones
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_avg_tanh.history["accuracy"], label="Train Acc AvgPool+tanh")
plt.plot(history_avg_tanh.history["val_accuracy"], label="Val Acc AvgPool+tanh")
plt.plot(history_max_relu.history["accuracy"], label="Train Acc MaxPool+ReLU")
plt.plot(history_max_relu.history["val_accuracy"], label="Val Acc MaxPool+ReLU")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history_avg_tanh.history["loss"], label="Train Loss AvgPool+tanh")
plt.plot(history_avg_tanh.history["val_loss"], label="Val Loss AvgPool+tanh")
plt.plot(history_max_relu.history["loss"], label="Train Loss MaxPool+ReLU")
plt.plot(history_max_relu.history["val_loss"], label="Val Loss MaxPool+ReLU")
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

