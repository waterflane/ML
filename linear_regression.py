import numpy as np
import matplotlib.pyplot as plt


def nn(theta0, theta1, x):
    res = y_pred = theta0 + theta1*x

    return res

def compute_err(y, y_pred):
    # print(y, y_pred)

    # print(np.sum((y - y_pred)**2)/len(y))
    # print(np.sum(abs((y - y_pred)))/len(y))

    err = np.sum((y - y_pred)**2)/len(y)
    max_err = np.max(np.abs(y - y_pred))

    return err, max_err

def compute_theta(x,y, x_mean, y_mean):
    theta1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    theta0 = y_mean - theta1*x_mean

    return theta0, theta1

def visualize(x, y, y_pred, y_mean):
    plt.scatter(x, y, label="Данные")
    plt.plot(x, y_pred, label="Регрессия")

    plt.axhline(y_mean, linestyle="--", label="y_mean")

    plt.legend()
    plt.show()

def tests(x,y,x_mean,y_mean):
    t0, t1 = compute_theta(x,y,x_mean,y_mean)
    y_pred = nn(t0, t1, x)

    err, max_err = compute_err(y, y_pred)
    print(f"Средняя ошибка: {err}")
    print(f"Максимальная ошибка: {max_err}")

    visualize(x, y, y_pred, y_mean)

def main():
    # x = np.array([1, 2, 3, 4])
    # y = np.array([2, 3, 5, 7])
    # x_mean= np.mean(x)
    # y_mean= np.mean(y)

    # tests(x,y,x_mean,y_mean)

    x = np.array([1, 2, 3, 4, 5 , 6])
    y = np.array([2, 3, 5, 7, 13, 15])
    x_mean= np.mean(x)
    y_mean= np.mean(y)

    tests(x,y,x_mean,y_mean)

if __name__ == "__main__":
    main()