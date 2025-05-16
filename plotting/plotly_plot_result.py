import sys
import git
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
from plotting.plotly_utils import (
    generate_random_color, create_box, create_box_outline,
    add_bin_lines, visual_setup
)
from stacking_objects.stack import Stack


def plotly_plot_stack(items, W, D, show=False, save_path=False, rng=829):

    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    # H = min(W, D)
    stack = Stack((D, W), 10_000)
    fig = go.Figure()
    colours = px.colors.qualitative.Pastel
    for i, item in enumerate(items):
        stack.place(item)
        x, y, z = item.get_position()
        dx, dy, dz = item.get_dimensions()
        color = colours[i % len(colours)]
        box = create_box(x, y, z, dx, dy, dz, color=color, opacity=1)
        # outline = create_box_outline(x, y, z, dx, dy, dz, color='black', width=3)
        fig.add_trace(box)
        # fig.add_trace(outline)
        # print(stack.height_map)
    H = stack.height()
    # add_bin_lines(fig, W, D, H)
    visual_setup(fig, W, D, H)

    if show:
        fig.show()
    if save_path:
        pio.write_image(fig, save_path, scale=2, width=1200, height=1000)
