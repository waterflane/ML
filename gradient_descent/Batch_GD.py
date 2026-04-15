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

def predict(weights, X):
    return X @ weights

def compute_loss(y_true, y_pred):
    return float(xp.mean((y_true - y_pred) ** 2))

def compute_gradient(X, y, weights):
    n = len(y)
    y_pred = X @ weights
    gradient = (2/n) * (X.T @ (y_pred - y))
    return gradient

def normalize(X): 
    mins = X.min(axis=0) 
    maxs = X.max(axis=0) 
    return (X-mins) / (maxs-mins + 1e-8), mins, maxs

def prepare_data(data):
    # полиномиальные признаки(Length^2, Diameter*Length), закоменьтировал тк они оказывается все портят
    # numeric_cols = ["Length", "Diameter", "Height", "Whole_Weight", "Shucked_Weight", "Viscera_Weight", "Shell_Weight"]
    # for col in numeric_cols:    data[f"{col}^2"] = data[col] ** 2
    # for c1, c2 in combinations(numeric_cols, 2):    data[f"{c1}*{c2}"] = data[c1] * data[c2]


    data = pd.get_dummies(data, columns=["Sex"], dtype=float)

    y = xp.asarray(data["Shell_Weight"].values.astype(float))
    X = xp.asarray(data.drop(columns=["Shell_Weight"]).values.astype(float))

    indices = xp.random.permutation(len(y))
    split = int(len(y)*0.9)

    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]

    X_train, mins, maxs = normalize(X_train)
    X_test = (X_test - mins) / (maxs - mins + 1e-8)

    X_train = xp.column_stack([xp.ones(len(X_train)), X_train])
    X_test = xp.column_stack([xp.ones(len(X_test)), X_test])

    return X_train, y_train, X_test, y_test

def evaluate(weights, X_test, y_test):
    y_pred = predict(weights, X_test)
    mse = compute_loss(y_test, y_pred)
    mae = float(xp.mean(xp.abs(y_test - y_pred)))

    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"{'Реальное'}     {'Предсказание'}      {'Ошибка'}")
    
    id = xp.random.choice(len(y_test), 15, replace=False)
    for i in id:
        err = float(y_pred[i] - y_test[i])
        print(f"{float(y_test[i]):>10.4f} {float(y_pred[i]):>14.4f} {err:>+10.4f}")

def plot_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()

def gradient_descent(X, y, lr=0.3981071705534972, epochs=1000): 
    # 0.3981071705534972 - оптимальный lr
    weights = xp.zeros(X.shape[1])
    loses = []

    for epoch in range(epochs):
        weights = weights - lr*compute_gradient(X, y, weights)
        y_pred = X @ weights

        loses.append(compute_loss(y, y_pred))
    
    return weights, loses

def find_optimal_lr(X_train, y_train, X_test, y_test):
    lrs = xp.logspace(-4, -0.4, 200)
    best_lr = float(lrs[0])
    best_mse = float("inf")
    results = []

    for lr in lrs:
        lr_val = float(lr)
        weights, _ = gradient_descent(X_train, y_train, lr=lr_val, epochs=1000)
        y_pred = predict(weights, X_test)
        mse = compute_loss(y_test, y_pred)

        if xp.isnan(xp.asarray(mse)) or xp.isinf(xp.asarray(mse)):
            mse = float("inf")

        results.append((lr_val, mse))
        if mse < best_mse:
            best_mse = mse
            best_lr = lr_val

    print(f"lr: {best_lr:.6f}  (MSE={best_mse:.4f})")

    valid = [(lr, mse) for lr, mse in results if mse < float("inf")]
    if valid:
        plt.figure()
        plt.plot([lr for lr, _ in valid], [mse for _, mse in valid], marker="o", markersize=3)
        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("MSE (test)")
        plt.title("Learning Rate vs MSE")
        plt.grid(True)
        plt.axvline(best_lr, color="r", linestyle="--", label=f"best={best_lr:.6f}")
        plt.legend()
        plt.show()

    return best_lr

def main():
    data = pd.read_csv("data/Class_Abalone.csv")

    X_train, y_train, X_test, y_test = prepare_data(data)

    # print(X_train[:5])
    # print(y_test[:5])
    # print(f"X_train: {X_train.shape}")
    # print(f"y_train: {y_train.shape}")
    # print(f"X_test:  {X_test.shape}")
    # print(f"y_test:  {y_test.shape}")

    # lr = find_optimal_lr(X_train, y_train, X_test, y_test)
    # print(lr)

    weights, loses = gradient_descent(X_train, y_train)

    evaluate(weights, X_test, y_test)
    plot_loss(loses)

if __name__ == "__main__":
    main()