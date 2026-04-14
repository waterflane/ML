import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def predict(weights, X):
    return X @ weights

def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def compute_gradient(X, y, weights):
    n = len(y)
    y_pred = X @ weights
    gradient = (2/n) * (X.T @ (y_pred-y))
    return gradient

def normalize(X):
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    return (X-mins) / (maxs-mins + 1e-8)

def prepare_data(data):
    data = pd.get_dummies(data, columns=["Sex"], dtype=float)

    y = data["Rings"].values.astype(float)
    X = data.drop(columns=["Rings"]).values.astype(float)

    indices = np.random.permutation(len(y))
    split = int(len(y)*0.9)

    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_test = np.column_stack([np.ones(len(X_test)), X_test])

    return X_train, y_train, X_test, y_test

def evaluate(weights, X_test, y_test):
    y_pred = predict(weights, X_test)
    mse = compute_loss(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))

    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"{'Реальное'}     {'Предсказание'}      {'Ошибка'}")
    
    id = np.random.choice(len(y_test), 10, replace=False)
    for i in id:
        err = y_pred[i]-y_test[i]
        print(f"{y_test[i]:>10.0f} {y_pred[i]:>14.2f} {err:>+10.2f}")

def plot_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()

def gradient_descent(X, y, lr=0.1, epochs=50):
    weights = np.zeros(X.shape[1])
    loses = []

    for epoch in range(epochs):
        weights = weights - lr*compute_gradient(X, y, weights)
        y_pred = X @ weights

        loses.append(compute_loss(y, y_pred))
    
    return weights, loses

def main():
    data = pd.read_csv("data/Class_Abalone.csv")

    X_train, y_train, X_test, y_test = prepare_data(data)

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")
    print(f"Первая строка X_train:{X_train[0]}")
    print(f"\nПервые 5 значений y_train: {y_train[:5]}")

    weights, loses = gradient_descent(X_train, y_train)

    plot_loss(loses)
    evaluate(weights, X_test, y_test)

if __name__ == "__main__":
    main()