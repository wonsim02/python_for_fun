from shapes import get_sphere, get_circle_xy, get_circle_xz, get_circle_yz, get_line, get_polygon
from graphs import get_surface, get_curve, get_xy_rectangle, colors, sqrt3

# default sphere
data = [
    get_surface(get_sphere(), colors.sphere_color, 0.5),

    get_curve(get_circle_xy(), colors.xy_color, width = 3),
    get_curve(get_circle_xz(), colors.xz_color, width = 3),
    get_curve(get_circle_yz(), colors.yz_color, width = 3),

    get_curve(get_line((-1, 0, 0), (1, 0, 0)), colors.color_black, width = 3),
    get_curve(get_line((0, -1, 0), (0, 1, 0)), colors.color_black, width = 3),
    get_curve(get_line((0, 0, -1), (0, 0, 1.25)), colors.color_black, width = 3),
]

# draws basic front-side rectangle
data.extend([
    get_curve(
        get_polygon((-sqrt3/2, 0, 0.5), (sqrt3/2, 0, 0.5), (sqrt3/2, 0, -1), (-sqrt3/2, 0, -1)),
        colors.xz_color, width = 3
    ),
    get_curve(get_line((-sqrt3/2, 0, -0.5), (sqrt3/2, 0, -0.5)), colors.color_black, width = 3),

])

# draws guidelines for ears and front-side faceline
data.extend([
    get_curve(get_circle_yz((-sqrt3/2, 0, -0.5), 0.5), colors.yz_color, width = 1),
    get_curve(get_circle_yz((sqrt3/2, 0, -0.5), 0.5), colors.yz_color, width = 1),

    get_curve(get_line((-sqrt3/2, -0.5, 0), (-sqrt3/3, -0.5, -1)), colors.color_black, width = 2),
    get_curve(get_line((sqrt3/2, -0.5, 0), (sqrt3/3, -0.5, -1)), colors.color_black, width = 2),

    get_curve(get_line((-sqrt3/2, 0.25, 0), (-sqrt3/3, 0.25, -1)), colors.color_black, width = 2),
    get_curve(get_line((sqrt3/2, 0.25, 0), (sqrt3/3, 0.25, -1)), colors.color_black, width = 2),

    get_curve(get_line((-sqrt3/3, -0.5, -1), (-sqrt3/3, 0.5, -1)), colors.color_black, width = 2),
    get_curve(get_line((sqrt3/3, -0.5, -1), (sqrt3/3, 0.5, -1)), colors.color_black, width = 2),

    get_curve(get_line((-sqrt3*5/12, -0.5, -0.5), (-sqrt3*5/12, 0.5, -0.5)), colors.color_black, width = 2),
    get_curve(get_line((sqrt3*5/12, -0.5, -0.5), (sqrt3*5/12, 0.5, -0.5)), colors.color_black, width = 2),
])

# draws guidelines for eyes and front-side faceline
data.extend([
    get_curve(get_xy_rectangle(-1), colors.xy_color, width = 2),
    get_curve(get_xy_rectangle(-0.5), colors.xy_color, width = 2),
    get_curve(get_xy_rectangle(0), colors.xy_color, width = 2),

    # get_curve(get_line((-sqrt3/2, -0.5, 0), (-sqrt3/2, -0.5, -1)), colors.xz_color, width = 1),
    # get_curve(get_line((sqrt3/2, -0.5, 0), (sqrt3/2, -0.5, -1)), colors.xz_color, width = 1),
])

# draws guidelines for nose and front-side faceline
data.extend([
    get_curve(
        get_polygon((0, -0.5, -1), (0, -0.5, 0.25), (0, -1, 0.25), (0, -1, -1)),
        colors.yz_color, width = 2
    ),
    get_curve(get_line((0, -1, -0.5), (0, 0, -0.5)), colors.yz_color, width = 1),
    get_curve(get_line((0, -0.75, 0.25), (0, -0.75, -1)), colors.yz_color, width = 1),
])
