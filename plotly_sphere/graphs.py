import plotly.graph_objects as go
import numpy as np

from shapes import get_polygon as _get_polygon


def get_surface(xyz, rgb, opacity):
    x_values, y_values, z_values = xyz

    assert(z_values.shape == x_values.shape)
    assert(z_values.shape == y_values.shape)

    color = f'rgb{rgb}'
    colorscale = [[0, color], [1, color]]
    color_level = np.zeros(z_values.shape)

    return go.Surface(
        x = x_values,
        y = y_values,
        z = z_values,
        showscale = False,
        colorscale = colorscale,
        surfacecolor = color_level,
        opacity = opacity,
    )


def get_curve(xyz, rgb, width):
    x_values, y_values, z_values = xyz

    assert(z_values.shape == x_values.shape)
    assert(z_values.shape == y_values.shape)

    color = f'rgb{rgb}'
    colorscale = [[0, color], [1, color]]
    color_level = np.zeros(z_values.shape)

    return go.Scatter3d(
        x = x_values,
        y = y_values,
        z = z_values,
        mode = 'lines',
        line = dict(
            width = width,
            color = color_level,
            colorscale = colorscale,
        )
    )

sphere_color = (204, 204, 204)
xy_color = (255, 0, 0)
xz_color = (0, 255, 0)
yz_color = (0, 0, 255)
color_black = (0, 0, 0)
sqrt3 = 3 ** 0.5

def get_xy_rectangle(z_coord):
    return _get_polygon(
        (-sqrt3/2, -0.5, z_coord),
        (sqrt3/2, -0.5, z_coord),
        (sqrt3/2, 0.5, z_coord),
        (-sqrt3/2, 0.5, z_coord)
    )

class _Colors:
    sphere_color = (204, 204, 204)
    xy_color = (255, 0, 0)
    xz_color = (0, 255, 0)
    yz_color = (0, 0, 255)
    color_black = (0, 0, 0)

colors = _Colors()
sqrt3 = 3 ** 0.5
