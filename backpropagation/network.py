import numpy
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd

class NeuralNetwork:
    def __init__(self, data, _ = numpy):
        global xp
        xp = _
        self.data = data
        self.X_train, self.y_train, self.X_test, self.y_test = self.prepare_data(data)

    def compute_loss(self, y_true, y_pred):
        y_true = xp.asarray(y_true).reshape(-1)
        y_pred = xp.asarray(y_pred).reshape(-1)
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
    
    
    def compute_y(self, X,  weights):
        first_l = X @ weights[0]
        out = first_l @ weights[1]
        return out.reshape(-1)

    def compute_gradient(self, weights, lr):
        X, y = self.X_train, self.y_train
        y = y.reshape(-1, 1)
        n = len(y)

        hidden = X @ weights[0]
        y_pred = hidden @ weights[1]
        dY = (2/n) * (y_pred - y)

        dW2 = hidden.T @ dY
        dHidden = dY @ weights[1].T
        dW1 = X.T @ dHidden

        weights[0] -= lr*dW1
        weights[1] -= lr*dW2

        return weights, y_pred.reshape(-1)
    
    def learning(self, lr=0.1, epochs=100):
        X, y = self.X_train, self.y_train

        n_input = X.shape[1]
        n_hidden = 6
        n_output = 1

        W1 = xp.random.randn(n_input, n_hidden) 
        W2 = xp.random.randn(n_hidden, n_output)

        weights = [W1, W2]
        loses = []

        for epoch in range(epochs):
            weights, y_pred = self.compute_gradient(weights, lr)
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