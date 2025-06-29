import sys
import git
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import colorcet
# CMAP = colorcet.glasbey
# CMAP = colorcet.glasbey_category10
CMAP = colorcet.glasbey_light
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
from plotting.pyvista_utils import *


def pyvista_plot_stack(
    items,
    W,
    D,
    show=False,
    save_path=False,
):

    pv.global_theme.color_cycler = 'default'
    plotter = pv.Plotter(window_size=(2000, 2000), off_screen=(not show))
    plotter.renderer.set_color_cycler(CMAP)
    for ix, item in enumerate(items):
        x, y, z = item.get_position()
        dx, dy, dz = item.get_dimensions()
        add_cube(
            plotter, x, y, z, dx, dy, dz,
            opacity=1, show_edges=False
        )

    if show:
        plotter.show()
    if save_path:
        plotter.save_graphic(save_path, raster=False, painter=True)
    plotter.close()


def pyvista_plot_many_stacks(
    items,
    W,
    D,
    shape,
    compactness=None,
    lbs=None,
    heights=None,
    n_to_stack=None,
    n_stacked=None,
    show=False,
    save_path=False,
):

    plotter = pv.Plotter(
        shape=shape,
        window_size=(2000, 500),
        off_screen=(not show),
        border=False
    )

    rows, cols = shape[0], shape[1]
    row_weights = []

    count = 0
    for row in range(rows):
        for col in range(cols):
            subitems = items[count]
            plotter.subplot(row, col)
            plotter.renderer.set_color_cycler(CMAP)

            if compactness is not None:
                plotter.add_text(
                    f'C ={compactness[count]}', font='courier',
                    color='k', font_size=20, position=(0, 0.9), viewport=True
                )

            if not any(e is None for e in (n_to_stack, n_stacked)):
                plotter.add_text(
                    f'N ={n_stacked[count]}/{n_to_stack[count]}',
                    font='courier', color='k', font_size=20,
                    position=(0, 0.8), viewport=True
                )

            if heights is not None:
                plotter.add_text(
                    f'H ={heights[count]}', font='courier', #orientation=90.0,
                    color='k', font_size=20,  # position='left_edge',
                    # position=(0.05, 0.5), viewport=True
                    position=(0, 0.7), viewport=True
                )

            if n_stacked is not None:
                include_up_to = n_stacked[count]
            else:
                include_up_to = len(subitems)

            for ix, item in enumerate(subitems[:include_up_to]):
                x, y, z = item.get_position()
                dx, dy, dz = item.get_dimensions()
                add_cube(
                    plotter, x, y, z, dx, dy, dz,
                    opacity=1, show_edges=False
                )

            if lbs is not None:
                box = pv.Box(bounds=(0, W, 0, D, 0, lbs[count]))
                plotter.add_mesh(
                    box, style='wireframe', color='black', line_width=1
                )
                plotter.add_text(
                    f'LB={lbs[count]}', font='courier', #orientation=90.0,
                    color='k', font_size=20,  # position='left_edge',
                    # position=(0.05, 0.5), viewport=True
                    position=(0, 0.6), viewport=True
                )
            plotter.camera.zoom(0.8)
            # plotter.camera.tight(padding=0.10)
            count += 1

    if show:
        plotter.show()
    if save_path:
        plotter.save_graphic(save_path, raster=False, painter=True)
    plotter.close()


def pyvista_make_movie(
    items,
    W,
    D,
    show=False,
    save_path=False,
):
    pv.global_theme.color_cycler = 'default'
    plotter = pv.Plotter(window_size=(2000, 2000), off_screen=(not show))
    plotter.open_movie(save_path)
    
    # Calculate scene bounds from all items first
    all_bounds = []
    for item in items:
        x, y, z = item.get_position()
        dx, dy, dz = item.get_dimensions()
        all_bounds.extend([
            [x - dx/2, x + dx/2],
            [y - dy/2, y + dy/2], 
            [z - dz/2, z + dz/2]
        ])
    
    # Set camera position based on scene bounds
    if all_bounds:
        x_range = [min(b[0] for b in all_bounds[::3]), max(b[1] for b in all_bounds[::3])]
        y_range = [min(b[0] for b in all_bounds[1::3]), max(b[1] for b in all_bounds[1::3])]
        z_range = [min(b[0] for b in all_bounds[2::3]), max(b[1] for b in all_bounds[2::3])]
        
        center = [(x_range[0] + x_range[1])/2, 
                  (y_range[0] + y_range[1])/2, 
                  (z_range[0] + z_range[1])/2]
        
        # Position camera at a reasonable distance
        max_range = max(x_range[1] - x_range[0], 
                       y_range[1] - y_range[0], 
                       z_range[1] - z_range[0])
        
        camera_distance = max_range * 2
        plotter.camera.position = (center[0] + camera_distance, 
                                  center[1] + camera_distance, 
                                  center[2] + camera_distance)
        plotter.camera.focal_point = center
        plotter.camera.up = (0, 0, 1)
    
    plotter.renderer.set_color_cycler(CMAP)
    plotter.write_frame()
    
    for ix, item in enumerate(items):
        x, y, z = item.get_position()
        dx, dy, dz = item.get_dimensions()
        add_cube(
            plotter, x, y, z, dx, dy, dz,
            opacity=1, show_edges=False
        )
        plotter.write_frame()
        plotter.write_frame()
        plotter.write_frame()
        plotter.write_frame()
    
    plotter.close()