import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as la
from sklearn.metrics import r2_score


# Define class for Least-Squares Regression
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
        self.k = np.shape(x)[1] + 1
        x = np.append(x, np.ones(np.shape(x)[0]).reshape(-1, 1), axis=1)
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

# Define class for our metric
def accuracy(a, b):
    residuals = np.array(a)-np.array(b)
    mean_error = np.mean(abs(residuals))
    return mean_error, max(abs(residuals))

# Generate 1000 non-stationary random walk process for each size and count t-statistics for each case
result_par = np.array([])
t = [25, 50, 75, 100, 150, 250, 350, 500, 600]
np.random.seed(7)
for T in t:
    t_calc_array_par = np.array([])
    for i in range(1000):
        '''
        Теперь 1000 раз сгенерируем различные процессы белого щума и по каждому построим процессс случайного блуждания.
        Для каждого такого процесса посчитаем необходимую t-сатистику для тестирования единичного корня, а затем посчитаем
        по этим расчетным хначениям их 5%-ный квартиль.
        '''
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
line1 = plt.plot(t, result_par, linewidth=1, label='parametric bootstrap', )
line2 = plt.plot([25, 50, 100, 250, 500, 600], [-3.00, -2.93, -2.89, -2.88, -2.87, -2.86], linewidth=1, label='table values', c='r')
plt.xlabel('Объём выборки (длина временного ряда)')
plt.ylabel('Критическое значение t-статитиски')
plt.title('Критические значения \n t-статистики для теста Дики-Фуллера (уровень значимости = 5%) \n c константой и без тренда \n'
          'для разных длин временного ряда')
ax = plt.gca()
ax.set_ylim([-4, -2])
plt.legend(loc=2)
plt.show()
plt.close()

result_par = list(result_par)
result_par.pop(2)
result_par.pop(4)
result_par.pop(6)
print(accuracy( [-3.00, -2.93, -2.89, -2.88, -2.87, -2.86], result_par))