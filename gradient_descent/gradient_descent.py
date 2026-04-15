import numpy
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd

# поменять USE_GPU на False, если надо на CPU
# нужно установить cupy-cuda11x или cupy-cuda12x и иметь видеокарту nvidia с поддержкой cuda
USE_GPU = False

if USE_GPU:
    import cupy
    xp = cupy
else:
    xp = numpy

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
    # полиномиальные признаки(Length^2, Diameter*Length), можно закоменьтировать тк роли особой не сыграло
    numeric_cols = ["Length", "Diameter", "Height", "Whole_Weight", "Shucked_Weight", "Viscera_Weight", "Shell_Weight"]
    for col in numeric_cols:    data[f"{col}^2"] = data[col] ** 2
    for c1, c2 in combinations(numeric_cols, 2):    data[f"{c1}*{c2}"] = data[c1] * data[c2]


    data = pd.get_dummies(data, columns=["Sex"], dtype=float)

    y = xp.asarray(data["Rings"].values.astype(float))
    X = xp.asarray(data.drop(columns=["Rings"]).values.astype(float))

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
        print(f"{float(y_test[i]):>10.0f} {float(y_pred[i]):>14.2f} {err:>+10.2f}")

def plot_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()

def gradient_descent(X, y, lr=0.08, epochs=1000): 
    # lr=0.08 оптимальное
    weights = xp.zeros(X.shape[1])
    loses = []

    for epoch in range(epochs):
        weights = weights - lr*compute_gradient(X, y, weights)
        y_pred = X @ weights

        loses.append(compute_loss(y, y_pred))
    
    return weights, loses

def main():
    data = pd.read_csv("data/Class_Abalone.csv")

    X_train, y_train, X_test, y_test = prepare_data(data)

    # print(f"X_train: {X_train.shape}")
    # print(f"y_train: {y_train.shape}")
    # print(f"X_test:  {X_test.shape}")
    # print(f"y_test:  {y_test.shape}")

    weights, loses = gradient_descent(X_train, y_train)

    evaluate(weights, X_test, y_test)
    plot_loss(loses)

if __name__ == "__main__":
    main()