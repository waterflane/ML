import numpy
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd

# поменять xp на cupy(раскомментировать ниже), если надо на видеокарте
# Для GPU (NVIDIA):
#   1. Установить CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
#   2. pip install cupy-cuda13x  (или cupy-cuda12x для CUDA 12)
#   3. Установить переменные среды (если CuPy не находит CUDA):
#      set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\<версия>
#      добавить в PATH: %CUDA_PATH%\bin\x64

import cupy
xp = cupy
# xp = numpy  # раскомментировать для CPU

from network import NeuralNetwork

def test_nn(weights, X_test, y_test, nn: NeuralNetwork):
    y_pred = nn.compute_y(X_test, weights)
    mse = nn.compute_loss(y_test, y_pred)
    mae = float(xp.mean(xp.abs(y_test - y_pred)))

    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"{'Реальное'}     {'Предсказание'}      {'Ошибка'}")
    
    id = xp.random.choice(len(y_test), 15, replace=False)
    for i in id:
        err = float(y_pred[i] - y_test[i])
        print(f"{float(y_test[i]):>10.4f} {float(y_pred[i]):>14.4f} {err:>+10.4f}")

def draw_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()


def main():
    data = pd.read_csv("data/Class_Abalone.csv")
    nn = NeuralNetwork(data, cupy)

    weights, loses = nn.learning(lr=0.001, epochs=1000)
    X_test, y_test = nn.X_test, nn.y_test

    test_nn(weights, X_test, y_test, nn)
    draw_loss(loses)

if __name__ == "__main__":
    main()