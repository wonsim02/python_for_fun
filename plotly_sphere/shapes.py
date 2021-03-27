import numpy as np

def get_sphere(center = (0, 0, 0), radius = 1):
    phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0 :2.0 * np.pi : 100j]
    center_x, center_y, center_z = center

    x_sphere = np.sin(phi) * np.cos(theta) + center_x
    y_sphere = np.sin(phi) * np.sin(theta) + center_y
    z_sphere = np.cos(phi) + center_z

    return (x_sphere, y_sphere, z_sphere)

def get_circle_xy(center = (0, 0, 0), radius = 1):
    t = np.linspace(0.0, 2.0 * np.pi, 100)
    center_x, center_y, center_z = center

    x_circle = radius * np.cos(t) + center_x
    y_circle = radius * np.sin(t) + center_y
    z_circle = center_z * np.ones(t.shape)

    return (x_circle, y_circle, z_circle)

def get_circle_xz(center = (0, 0, 0), radius = 1):
    t = np.linspace(0.0, 2.0 * np.pi, 100)
    center_x, center_y, center_z = center

    x_circle = radius * np.cos(t) + center_x
    y_circle = center_y * np.ones(t.shape)
    z_circle = radius * np.sin(t) + center_z

    return (x_circle, y_circle, z_circle)

def get_circle_yz(center = (0, 0, 0), radius = 1):
    t = np.linspace(0.0, 2.0 * np.pi, 100)
    center_x, center_y, center_z = center

    x_circle = center_x * np.ones(t.shape)
    y_circle = radius * np.cos(t) + center_y
    z_circle = radius * np.sin(t) + center_z

    return (x_circle, y_circle, z_circle)

def get_line(pt1, pt2):
    x1, y1, z1 = pt1
    x2, y2, z2 = pt2

    return (np.array((x1, x2)), np.array((y1, y2)), np.array((z1, z2)))

def get_polygon(*pt):
    start_pt = pt[0]
    return tuple(map(np.array, zip(*pt, start_pt)))
