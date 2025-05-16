import pyvista as pv


def add_cube(
    plotter,
    x, y, z,
    dx, dy, dz,
    opacity=1,
    show_edges=True
):
    center = (x + dx/2.0, y + dy/2.0, z + dz/2.0)
    cube_mesh = pv.Cube(center=center, x_length=dx, y_length=dy, z_length=dz)
    plotter.add_mesh(
        cube_mesh,
        opacity=opacity,
        show_edges=show_edges,
    )
