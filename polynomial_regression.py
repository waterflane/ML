import numpy as np
import matplotlib.pyplot as plt

def nn(theta_list, x):
    res = np.sum([theta_list[i] * x**i for i in range(len(theta_list))], axis=0)

    return res

def compute_err(y, y_pred):
    err = np.sum((y - y_pred)**2)/len(y)
    max_err = np.max(np.abs(y - y_pred))

    return err, max_err

def compute_theta(x,y, x_mean, y_mean):
    X = np.column_stack([np.ones(len(x)), x, x**2])

    theta = np.linalg.lstsq(X, y, rcond=None)[0]

    print(theta)
    return theta

def visualize(x, y, y_pred, y_mean, theta_list):
    plt.scatter(x, y, label="Данные")

    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_smooth = nn(theta_list, x_smooth)
    plt.plot(x_smooth, y_smooth, label="Регрессия")

    plt.axhline(y_mean, linestyle="--", label="y_mean")
    plt.legend()
    plt.show()

def tests(x,y,x_mean,y_mean):
    t_list = compute_theta(x,y,x_mean,y_mean)
    y_pred = nn(t_list, x)

    err, max_err = compute_err(y, y_pred)
    print(f"Средняя ошибка: {err}")
    print(f"Максимальная ошибка: {max_err}")

    visualize(x, y, y_pred, y_mean, t_list)

def main():
    x = np.array([1, 2, 3, 4, 5 , 6])
    y = np.array([2, 3, 6, 10, 20, 29])
    x_mean= np.mean(x)
    y_mean= np.mean(y)
    # degree = 2

    tests(x,y, x_mean,y_mean)

if __name__ == "__main__":
    main()