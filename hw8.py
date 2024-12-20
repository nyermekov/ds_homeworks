from random import seed
import numpy as np

data = np.array([
    [50, 12, 175, 70, 0, 1],
    [60, 16, 180, 75, 0, 1],
    [55, 14, 165, 60, 1, 1],
    [75, 18, 185, 80, 0, 1],
    [48, 12, 170, 68, 0, 1],
    [65, 15, 175, 72, 1, 1],
    [72, 17, 178, 76, 0, 1],
    [52, 13, 162, 65, 1, 1],
    [80, 20, 190, 85, 0, 1],
    [58, 14, 168, 70, 1, 1],
    [70, 16, 177, 74, 0, 1],
    [45, 10, 160, 58, 1, 1],
    [90, 22, 192, 88, 0, 1],
    [53, 13, 171, 69, 1, 1],
    [62, 15, 173, 71, 0, 1],
    [48, 11, 166, 67, 1, 1],
    [85, 19, 188, 82, 0, 1],
    [55, 14, 169, 70, 1, 1],
    [75, 17, 176, 75, 0, 1],
    [43, 10, 162, 57, 1, 1]
])

Y = data[:, 0]
#coef2 * x2(гласная-согласная)
X2 = data[:, 2]
#coef3 * x3
X5 = data[:, 5]

seed(2024)
np.random.seed(2024)

iterations = 10000
learning_rate = 0.00001

coef_2 = np.random.rand()
coef_5 = np.random.rand()

n = len(Y)
for _ in range(iterations):
    Y_pred = coef_2 * X2 + coef_5 * X5

    d_coef_2 = (-2/n) * np.sum((Y - Y_pred) * X2)
    d_coef_5 = (-2/n) * np.sum((Y - Y_pred) * X5)

    coef_2 -= learning_rate * d_coef_2
    coef_5 -= learning_rate * d_coef_5

Y_final = coef_2 * X2 + coef_5 * X5
rmse = np.sqrt(np.mean((Y - Y_final) ** 2))

[[coef_2, coef_5], rmse]