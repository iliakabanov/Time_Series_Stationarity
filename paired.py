import numpy as np
import pandas as pd
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

# Generate 1000 non-stationary random walk process for size 25 and count t-statistics
# Using paired bootstrap
result = np.array([])
t = [25]
for T in t:
    # Сгенерируем процесс случайного блуждания для изначальной выборки
    original_y = np.array([0])
    original_noise = np.random.normal(size=T+1, loc=np.zeros(T+1), scale=np.ones(T+1))
    for j in range(1, T+1):
        original_y = np.append(original_y, original_noise[j] + original_y[j-1])
    original_delta_y = original_y[1:] - original_y[:-1]
    original_y = original_y[:-1]
    # Проведем регрессию delta_y(t) на y(t-1) для изначальной выборки
    paired_bootstrap = LinearRegression()
    paired_bootstrap.fit(x=original_y.reshape(-1, 1), y_true=original_delta_y.reshape(-1, 1))
    t_calc_array_paired = np.array([])
    estimates = np.array([])
    # Выведем оценку коэффициента при y(t-1) и её стандартную ошибку


    # Теперь 1000 раз отберем бутстреп-выборки с вовзращением для остатков, полученных в оригинальной выборке,
    # и получим 1000 новых бустреп-выборок (delta_y(t), y(t-1)), для которых будем считать t-статистики

    for i in range(1000):
        indexes = np.random.choice(T-1, size=T, replace=True)
        boot_y = np.array([])
        boot_delta_y = np.array([])
        for index in indexes:
            boot_y = np.append(boot_y, original_y[index])
            boot_delta_y = np.append(boot_delta_y, original_delta_y[index])
        b = LinearRegression()
        b.fit(x=boot_y.reshape(-1, 1), y_true=boot_delta_y.reshape(-1, 1))
        t_calc_paired = (b.coef_[0] - paired_bootstrap.coef_[0]) / b.se_[0]
        t_calc_array_paired = np.append(t_calc_array_paired, t_calc_paired)

# Посчитаем для полученных t-расчетных 5%-ный квантиль
result = np.append(result, np.quantile(t_calc_array_paired, 0.975))
print(result)

# Наглядно представим распределение t-расчетных
plt.figure(figsize=(7, 6))
plt.hist(t_calc_array_paired, bins=100)
plt.ylabel('Значения эмпирических расчетных t-статистик')
plt.xlabel('Количество выборок')
plt.title('Распределение расчетных t-статистик для выборок, \nполученных остаточным бустрепированием соотвествующей \n модели парной регрессии (T=25)')
# plt.show()
plt.close()

# Generate 1000 non-stationary random walk process for size 25 and count t-statistics
result = np.array([])
np.random.seed(2)
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
        regressors = y.reshape(-1, 1)
        parametric_bootstrap.fit(x=regressors, y_true=delta_y.reshape(-1, 1))
        t_calc_par = parametric_bootstrap.coef_[0] / parametric_bootstrap.se_[0]
        t_calc_array_par = np.append(t_calc_array_par, t_calc_par)
    result = np.append(result, np.quantile(t_calc_array_par, 0.05))

# Наглядно представим распределение t-расчетных
plt.figure(figsize=(7, 6))
plt.hist(t_calc_array_par, bins=100)
plt.axvline(result, ymin=0, ymax=35, color='red', label='5% quantile ({0:.2f})'.format(float(result)))
plt.legend(loc='upper right')
plt.ylabel('Количество выборок')
plt.xlabel('Значения эмпирических расчетных t-статистик')
plt.title('Эмпирическое распределение расчетных t-статистик \n для DF-теста c константой и трендом (T=25)')
plt.show()