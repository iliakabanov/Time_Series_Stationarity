import numpy as np
import random
import matplotlib.pyplot as plt
import numpy.linalg as la


# Пишем класс для использования стандартного метода наименьших квадратов
class LinearRegression:
    def __init__(self, **kwargs):
        self.coef_ = None
        self.s_square = None
        self.cov_matrix = None
        self.se_ = None
        self.n = None
        self.k = None
        self.residuals = None

    def fit(self, x: np.array, y_true: np.array):
        self.n = np.shape(y_true)[0]
        self.k = np.shape(x)[1] + 1
        x = np.append(x, np.ones(np.shape(x)[0]).reshape(-1, 1), axis=1)
        self.coef_ = np.matmul(la.inv(np.matmul(np.transpose(x), x)), np.matmul(np.transpose(x), y_true).reshape(-1, 1))
        y_pred = np.array(np.matmul(x, self.coef_).reshape(-1, 1))
        self.residuals_ = (y_true-y_pred)
        sum_sq_errors = sum(self.residuals_*self.residuals_)
        TSS = sum((y_true-np.mean(y_true))*(y_true-np.mean(y_true)))
        self.s_square = sum_sq_errors / (self.n - self.k)
        self.cov_matrix_ = la.inv(np.matmul(np.transpose(x), x))*self.s_square
        self.se_ = np.diagonal(self.cov_matrix_).reshape(-1, 1)**(1/2)

    def predict(self, x: np.array):
        x = np.append(x, np.ones(np.shape(x)[0]).reshape(-1, 1), axis=1)
        predictions = np.array(np.matmul(x, self.coef_).reshape(-1, 1))
        return predictions


# Generate 1000 non-stationary random walk process for each size and count t-statistics for each case
# Используем блочный бустреп
result = np.array([])
t_2 = np.array([25, 50, 75, 100, 150, 250, 350, 500, 600])*2
np.random.seed(3)
for T in t_2:
    # Сгенерируем процесс случайного блуждания для изначальной выборки
    original_y = np.array([0])
    original_noise = np.random.normal(size=T+1, loc=np.zeros(T+1), scale=np.ones(T+1))
    for j in range(1, T+1):
        original_y = np.append(original_y, original_noise[j] + original_y[j-1])
    original_delta_y = original_y[1:] - original_y[:-1]
    t_calc_array_blocked = np.array([])

    # Теперь 1000 раз отберем бутстреп-выборки с вовзращением для остатков, полученных в оригинальной выборке,
    # и получим 1000 новых бустреп-выборок (delta_y(t), y(t-1)), для которых будем считать t-статистики
    for i in range(1000):
        start = np.random.randint(T/3-1)
        n = np.random.randint(4, 46)
        boot_y = original_y[start:start+n]
        boot_delta_y = boot_y[1:] - boot_y[:-1]
        b = LinearRegression()
        b.fit(x=boot_y[:-1].reshape(-1, 1), y_true=boot_delta_y.reshape(-1, 1))
        t_calc_blocked = (b.coef_[0]) / b.se_[0]
        t_calc_array_blocked = np.append(t_calc_array_blocked, t_calc_blocked)
    result = np.append(result, np.quantile(t_calc_array_blocked, 0.05))


# Построим на графике такие 5_ные квартили для каждой длины временного ряда,а также их теоритические значения из статистики Дикки-Фуллера
plt.figure(figsize=(7, 6))
line1 = plt.plot(t_2/2, result, linewidth=1, label='parametric bootstrap')
line2 = plt.plot([25, 50, 100, 250, 500, 600], [-3, -2.93, -2.89, -2.88, -2.87, -2.86], linewidth=1, label='table values', c='r')
plt.xlabel('Объём выборки (длина временного ряда)')
plt.ylabel('Значение t-статитиски')
plt.title('Критические значения \n t-статистики для теста Дики-Фуллера (уровень значимости = 5%) \n c константой и без тренда \n'
          'для разных длин временного ряда')
ax = plt.gca()
ax.set_ylim([-4, -2])
plt.legend(loc=2)
plt.show()
plt.close()
