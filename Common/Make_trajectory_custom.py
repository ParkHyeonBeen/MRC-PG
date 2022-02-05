import numpy as np
import math
import matplotlib.pyplot as plt

path = '/temp/cocelRL-master/Etc/path_logger/'

points = np.array([0., 0.])

for i in range(199):
    points = np.vstack((points, np.array([0., 0.])))

Ys1to2 = np.linspace(0, 4, 1000)
Xs1to2 = 3*Ys1to2
for x, y in zip(Xs1to2, Ys1to2):
    points = np.vstack((points, np.array([x, y])))

Xs2to3 = np.linspace(12, 12.5, 300)
Ys2to3 = np.linspace(4, 5, 300)
for x, y in zip(Xs2to3, Ys2to3):
    points = np.vstack((points, np.array([x, y])))

steps = 500

for i in range(steps):
    x = 2.5 * math.cos((math.pi / steps) * i)
    y = 2.5 * math.sin((math.pi / steps) * i)
    points = np.vstack((points, np.array([x + 10, y + 5])))

for i in range(steps):
    x = 2.5 * math.cos(-(math.pi / steps) * i)
    y = 2.5 * math.sin(-(math.pi / steps) * i)
    points = np.vstack((points, np.array([x + 5, y + 5])))

for i in range(steps):
    x = 2.5 * math.cos((math.pi / steps) * i)
    y = 2.5 * math.sin((math.pi / steps) * i)
    points = np.vstack((points, np.array([x + 0, y + 5])))

for i in range(steps):
    x = 2.5 * math.cos(-(math.pi / steps) * i)
    y = 2.5 * math.sin(-(math.pi / steps) * i)
    points = np.vstack((points, np.array([x - 5, y + 5])))

for i in range(steps):
    x = 2.5 * math.cos((math.pi / steps) * i)
    y = 2.5 * math.sin((math.pi / steps) * i)
    points = np.vstack((points, np.array([x - 10, y + 5])))

Xs8to9 = np.linspace(-12.5, -5, 2000)
Ys8to9 = np.linspace(5, -7, 2000)
for x, y in zip(Xs8to9, Ys8to9):
    points = np.vstack((points, np.array([x, y])))

steps2 = 2000

for i in range(steps2):
    x = 5 * math.sin((math.pi / (steps2/2)) * i)
    y = 5 * math.cos((math.pi / (steps2/2)) * i)
    points = np.vstack((points, np.array([x - 5, y - 12])))

for i in range(100):
    points = np.vstack((points, np.array([0., 0.])))

plt.plot(points[:,0], points[:,1])
plt.show()

np.save(path + 'path', points)
print(len(points))

# path = 'C:/Users/owner/Desktop/Workspace_paper/temp/cocelRL-master/xml_background/path.npy'
# goal = np.load(path)
# for i in range(len(goal)):
#     print(goal[i])