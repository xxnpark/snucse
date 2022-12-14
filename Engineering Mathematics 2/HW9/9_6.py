from statistics import NormalDist
import numpy as np
import matplotlib.pyplot as plt

N = 10000

plt.title("Exercise 9.6")
plt.xlabel("z")
plt.ylabel("|P(X <= Z) - P(Y <= Z)|")

for z in np.linspace(-3, 3, 1000):
    x_le_z_count = 0
    for _ in range(N):
        x_le_z_count += int((sum(np.random.uniform(0, 1, 12)) - 6) <= z)
    p_x_le_z = x_le_z_count / N
    p_y_le_z = NormalDist(0, 1).cdf(z)
    plt.scatter(z, abs(p_x_le_z - p_y_le_z), color='black')

plt.savefig("plot.png")