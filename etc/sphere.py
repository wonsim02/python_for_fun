import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

sphere = (0.8, 0.8, 0.8, 0.5)
xy = (1, 0, 0, 1)
xz = (0, 1, 0, 1)
yz = (0, 0, 1, 1)

t = np.linspace(0, 2*np.pi, 100)
sqrt3 = 3**0.5

circle_xy = lambda px, py, pz, r: (r*np.cos(t)+px, r*np.sin(t)+py, 0*t+pz)
circle_xz = lambda px, py, pz, r: (r*np.cos(t)+px, 0*t+py, r*np.sin(t)+pz)
circle_yz = lambda px, py, pz, r: (0*t+px, r*np.cos(t)+py, r*np.sin(t)+pz)
line = lambda pt1, pt2: tuple(zip(pt1, pt2))
poly = lambda *pt: tuple(zip(*pt, pt[0]))

def draw(shape, linewidth, color):
    ax.plot(*shape, linewidth = linewidth, color = color)

# Get instance of Axis3D
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
plt.axis('off')

phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
x = np.sin(phi)*np.cos(theta)
y = np.sin(phi)*np.sin(theta)
z = np.cos(phi)

ax.plot_surface(x, y, z, color = sphere)
ax.scatter([1, 0, 0], [0, -1, 0], [0, 0, 1], color = (0, 0, 0, 1), linewidth = 5)

draw(circle_xy(0, 0, 0, 1), 3, xy)
draw(circle_xz(0, 0, 0, 1), 3, xz)
draw(circle_yz(0, 0, 0, 1), 3, yz)

draw(line((-1, 0, 0), (1, 0, 0)), 3, (0, 0, 0, 0.5))
draw(line((0, -1, 0), (0, 1, 0)), 3, (0, 0, 0, 0.5))
draw(line((0, 0, -1), (0, 0, 1)), 3, (0, 0, 0, 0.5))

draw(poly(
    (-sqrt3/2, 0, 0.5), (sqrt3/2, 0, 0.5), 
    (sqrt3/2, 0, -1), (-sqrt3/2, 0, -1)), 3, xz)
draw(line((-sqrt3/2, 0, -0.5), (sqrt3/2, 0, -0.5)), 3, xz)

draw(circle_yz(-3**0.5/2, 0, -0.5, 0.5), 1, yz)
draw(circle_yz(3**0.5/2, 0, -0.5, 0.5), 1, yz)

for zcoord in [-1, -0.5, 0]:
    draw(poly(
        (-sqrt3/2, -0.5, zcoord), (sqrt3/2, -0.5, zcoord), 
        (sqrt3/2, 0.5, zcoord), (-sqrt3/2, 0.5, zcoord)), 2, xy)

draw(line((-3**0.5/2, -0.5, 0), (-3**0.5/2, -0.5, -1)), 1, xz)
draw(line((3**0.5/2, -0.5, 0), (3**0.5/2, -0.5, -1)), 1, xz)

draw(poly((0, -0.5, -1), (0, -0.5, 0.25), (0, -1, 0.25), (0, -1, -1)), 2, yz)
draw(line((0, -1, -0.5), (0, 0, -0.5)), 1, yz)
draw(line((0, -0.75, 0.25), (0, -0.75, -1)), 1, yz)

plt.show()