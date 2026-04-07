import numpy as np

# данные
x = np.array([1, 2, 3, 4])
y = np.array([2, 3, 5, 7])

# средние
x_mean = np.mean(x)
y_mean = np.mean(y)

# считаем θ1 (наклон)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)

theta1 = numerator / denominator

# считаем θ0 (сдвиг)
theta0 = y_mean - theta1 * x_mean

print("theta1 =", theta1)
print("theta0 =", theta0)

y_pred = theta0 + theta1 * x
print(y_pred)

import matplotlib.pyplot as plt

plt.scatter(x, y, label="Данные")
plt.plot(x, y_pred, label="Регрессия")

plt.axhline(y_mean, linestyle="--", label="y_mean")

plt.legend()
plt.show()