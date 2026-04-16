import numpy
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd
from main import xp

class NeuralNetwork:
    def __init__(self, data):
        self.data = data
        self.X_train, self.y_train, self.X_test, self.y_testself.prepare_data(data)

    def predict(self, weights, X):
        return X @ weights
    def compute_loss(self, y_true, y_pred):
        return float(xp.mean((y_true - y_pred) ** 2))
    

    def normalize(self, X): 
        mins = X.min(axis=0) 
        maxs = X.max(axis=0) 
        return (X-mins) / (maxs-mins + 1e-8), mins, maxs
    def prepare_data(self, data):
        data = pd.get_dummies(data, columns=["Sex"], dtype=float)

        y = xp.asarray(data["Rings"].values.astype(float))
        X = xp.asarray(data.drop(columns=["Rings"]).values.astype(float))

        indices = xp.random.permutation(len(y))
        split = int(len(y)*0.9)

        X_train, X_test = X[indices[:split]], X[indices[split:]]
        y_train, y_test = y[indices[:split]], y[indices[split:]]

        X_train, mins, maxs = self.normalize(X_train)
        X_test = (X_test-mins) / (maxs-mins + 1e-8)

        X_train = xp.column_stack([xp.ones(len(X_train)), X_train])
        X_test = xp.column_stack([xp.ones(len(X_test)), X_test])

        return X_train, y_train, X_test, y_test
    

    def compute_gradient(self, weights):
        X, y = self.X_train, self.y_train

        n = len(y)
        y_pred = X @ weights
        gradient = (2/n) * (X.T @ (y_pred - y))
        return gradient
    
    def learning(self, lr=0.1, epochs=100):
        X, y = self.X_train, self.y_train
        
        weights = xp.zeros(X.shape[1])
        loses = []

        for epoch in range(epochs):
            weights = weights - lr*self.compute_gradient(X, y, weights)
            y_pred = X @ weights

            loses.append(self.compute_loss(y, y_pred))
        
        return weights, loses


    def find_optimal_lr(self):
        X_train, y_train, X_test, y_test = self.X_train, self.y_train, self.X_test, self.y_test

        lrs = xp.logspace(-4, -0.4, 200)
        best_lr = float(lrs[0])
        best_mse = float("inf")
        results = []

        for lr in lrs:
            lr_val = float(lr)
            weights, _ = self.gradient_descent(X_train, y_train, lr=lr_val, epochs=1000)
            y_pred = self.predict(weights, X_test)
            mse = self.compute_loss(y_test, y_pred)

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