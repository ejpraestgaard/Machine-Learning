import numpy as np
import matplotlib.pyplot as plt
import random
import math

#first, generate two points to cluster around:
center_0 = np.array([random.uniform(0, 5), random.uniform(0,5)])
center_1 = np.array([random.uniform(5, 10), random.uniform(5, 10)])

#then generate the cluters. x is the center, r is the radius, and n is the number of points. 
#I am generating the cloud as 2d points because I think it could be pedagogically usefull.
def cloud(c, r = 5, n = 100):
    rho = np.random.uniform(0, r, n)
    theta = np.random.uniform(0, 2*math.pi, n)
    
    xs = [math.cos(theta[i]) for i in range(n)]
    ys = [math.sin(theta[i]) for i in range(n)]
    
    points = np.array([rho * xs, rho * ys])
    cloud = [points.T[i] + c for i in range(n)]
    return cloud

cloud1 = cloud(center_0)
cloud2 = cloud(center_1)

x = np.linspace(0, 10)

x_0s = [cloud1[i][0] for i in range(len(cloud1))]
y_0s = [cloud1[i][1] for i in range(len(cloud1))]

x_1s = [cloud2[i][0] for i in range(len(cloud1))]
y_1s = [cloud2[i][1] for i in range(len(cloud1))]

plt.scatter(x_0s, y_0s, label = 'Cloud 1')
plt.scatter(x_1s, y_1s, label = 'Cloud 2')
plt.legend()
plt.show()
