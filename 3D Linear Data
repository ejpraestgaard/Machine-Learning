import numpy as np
import matplotlib.pyplot as plt

#function which makes the data
def random_linear_data(dom = 20, n = 100, noise = 5):
    x = np.random.uniform(-dom, dom, size = (n,))
    y = np.random.uniform(-dom, dom, size = (n,))
    coeff_0 = np.random.uniform(-5, 5)
    coeff_1 = np.random.uniform(-5, 5)
    z = []
    for i in range(n):
        epsilon = np.random.uniform(-noise, noise)
        z.append((coeff_0 + epsilon)*x[i] + (coeff_1 + epsilon)*y[i])
    data = [[x[i], y[i], z[i]] for i in range(n)]
    return x, y, z


#plotting the data
x, y, z = random_linear_data()

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(x,y,z)
plt.show()
