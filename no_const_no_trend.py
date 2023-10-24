import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as la
from itertools import combinations_with_replacement


class LinearRegression:
    def __init__(self, **kwargs):
        self.coef_ = None
        self.s_square = None
        self.cov_matrix = None
        self.se_ = None
        self.n = None
        self.k = None
        self.residuals = None
        pass

    def fit(self, x: np.array, y_true: np.array):
        self.n = np.shape(y_true)[0]
        self.k = np.shape(x)[1]
        self.coef_ = np.matmul(la.inv(np.matmul(np.transpose(x), x)), np.matmul(np.transpose(x), y_true).reshape(-1, 1))
        y_pred = np.array(np.matmul(x, self.coef_).reshape(-1, 1))
        self.residuals = (y_true-y_pred)
        sum_sq_errors = sum(self.residuals*self.residuals)
        TSS = sum((y_true-np.mean(y_true))*(y_true-np.mean(y_true)))
        self.s_square = sum_sq_errors / (self.n - self.k)
        self.cov_matrix_ = la.inv(np.matmul(np.transpose(x), x))*self.s_square
        self.se_ = np.diagonal(self.cov_matrix_).reshape(-1, 1)**(1/2)

    def predict(self, x: np.array):
        x = np.append(x, np.ones(np.shape(x)[0]).reshape(-1, 1), axis=1)
        predictions = np.array(np.matmul(x, self.coef_).reshape(-1, 1))
        return predictions


def accuracy(a, b):
    residuals = np.array(a)-np.array(b)
    mean_error = np.mean(abs(residuals))
    return mean_error, max(abs(residuals))

# Generate 1000 non-stationary random walk process for each size and count t-statistics for each case
result_par = np.array([])
t = [25, 50, 75, 100, 150, 250, 350, 500, 600]
np.random.seed(10)
for T in t:
    t_calc_array_par = np.array([])
    for i in range(1000):
        y = np.array([0])
        white_noise = np.random.normal(size=T+1, loc=np.zeros(T+1), scale=np.ones(T+1))
        for j in range(1, T+1):
            y = np.append(y, white_noise[j] + y[j-1])
        delta_y = y[1:] - y[:-1]
        y = y[:-1]
        parametric_bootstrap = LinearRegression()
        parametric_bootstrap.fit(x=y.reshape(-1, 1), y_true=delta_y.reshape(-1, 1))
        t_calc_par = parametric_bootstrap.coef_[0] / parametric_bootstrap.se_[0]
        t_calc_array_par = np.append(t_calc_array_par, t_calc_par)
    result_par = np.append(result_par, np.quantile(t_calc_array_par, 0.05))

# Create graphics of empirical and theoretical t-statistics critical values
plt.figure(figsize=(8, 7))
line1 = plt.plot(t, result_par, linewidth=1, label='parametric bootstrap', c='b')
line2 = plt.plot([25, 50, 100, 250, 500, 600], [-1.95, -1.95, -1.95, -1.95, -1.95, -1.95], linewidth=1, label='table values', c='r')
plt.xlabel('Объём выборки (длина временного ряда)')
plt.ylabel('Критическое значение t-статитиски')
plt.title('Критические значения \n t-статистики для теста Дики-Фуллера (уровень значимости = 5%) \n без константы и без тренда \n'
          'для разных длин временного ряда')
ax = plt.gca()
ax.set_ylim([-2.25, -1.65])
plt.legend(loc=2)
plt.show()
plt.close()

# Create graphics of empirical and theoretical t-statistics critical values
plt.figure(figsize=(8, 7))
line1 = plt.plot(t, result_par, linewidth=1, label='parametric bootstrap', c='b')
line2 = plt.plot([25, 50, 100, 250, 500, 600], [-2.66, -2.62, -2.6, -2.58, -2.58, -2.58], linewidth=1, label='table values', c='r')
plt.xlabel('Объём выборки (длина временного ряда)')
plt.ylabel('Критическое значение t-статитиски')
plt.title('Критические значения \n t-статистики для теста Дики-Фуллера (уровень значимости = 1%) \n без константы и без тренда \n'
          'для разных длин временного ряда')
ax = plt.gca()
ax.set_ylim([-2.9, -2.3])
plt.legend(loc=2)
plt.show()
plt.close()


