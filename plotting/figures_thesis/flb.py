import sys
import git
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
from plotting.plotly_utils import (
    generate_random_color,
    create_box,
    create_box_outline,
    add_bin_lines,
    visual_setup,
    visual_setup_2,
)
from stacking_objects.item import Item
from stacking_objects.stack import Stack


W = 10
D = 10
H = 10


def plot_stuff(fig, items, **trace_kwargs):

    # stack = Stack((D, W), 10_000)

    for item in items:
        x, y, z = item.get_position()
        dx, dy, dz = item.get_dimensions()
        box = create_box(x, y, z, dx, dy, dz, color='grey', opacity=1)
        fig.add_trace(box, **trace_kwargs)
        outline = create_box_outline(x, y, z, dx, dy, dz, color='black', width=3)
        fig.add_trace(outline, **trace_kwargs)

    # fig.add_trace(go.Scatter3d(x=[6], y=[4], z=[0.04], mode='markers', marker=dict(size=14, color='red')))

    add_bin_lines(fig, W, D, H)
    # visual_setup(fig, W, D, H, showgrid=True, showticks=True, x_title='x', y_title='y', z_title='z')

    # for item in items:
    #     # since 0.01 may be added to increase visibility
    #     item.x, item.y, item.z = np.asarray(item.get_position(), dtype=int)
    #     stack.place(item)


def plot_box_right(fig, items, **trace_kwargs):
    for item in items:
        x, y, z = item.get_position()
        dx, dy, dz = item.get_dimensions()
        box = create_box(x, y, z, dx, dy, dz, color='grey', opacity=1)
        fig.add_trace(box, **trace_kwargs)
        outline = create_box_outline(x, y, z, dx, dy, dz, color='black', width=3)
        fig.add_trace(outline, **trace_kwargs)

    # fig.add_trace(go.Scatter3d(x=[6], y=[4], z=[0.04], mode='markers', marker=dict(size=14, color='red')))

    add_bin_lines(fig, W, D, H, **trace_kwargs)
    # visual_setup_2(fig, W, D, H, showgrid=True, showticks=True, x_title='x', y_title='y', z_title='z')


fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}]],
    subplot_titles=['Pallet Items', 'Upcoming Item'],
    horizontal_spacing=0  # tighten the space between the plots
)

# Example pallet dimensions.
W, D, H = 10, 10, 10

# Define current pallet items.
item_1 = Item((4, 4, 4), 0)
item_1.x, item_1.y, item_1.z = 0, 0, 0

item_2 = Item((2, 2, 2), 0)
item_2.x, item_2.y, item_2.z = 0, 0, 4.03

item_3 = Item((3, 3, 3), 0)
item_3.x, item_3.y, item_3.z = 0, 7, 0

item_4 = Item((4, 4, 4), 0)
item_4.x, item_4.y, item_4.z = 6, 0, 0

current_items = [item_1, item_2, item_3, item_4]

# Add pallet items (scene in the first subplot: col 1)
for item in current_items:
    x, y, z = item.get_position()
    dx, dy, dz = item.get_dimensions()
    # Add the solid box.
    box_trace = create_box(x, y, z, dx, dy, dz, color='grey', opacity=0)
    fig.add_trace(box_trace, row=1, col=1)
    # Add the outline.
    outline_trace = create_box_outline(x, y, z, dx, dy, dz, color='black', width=3)
    fig.add_trace(outline_trace, row=1, col=1)

# Add bin boundary lines to the first scene.
add_bin_lines(fig, row=1, col=1, W=W, D=D, H=H)

# Define the upcoming item.
upcoming_item = Item((4, 4, 4), 0)
upcoming_item.x, upcoming_item.y, upcoming_item.z = 0, 0, 0  # coordinate in its own system

# Add upcoming item in the second subplot (scene on the right).
x, y, z = upcoming_item.get_position()
dx, dy, dz = upcoming_item.get_dimensions()
# Draw the upcoming item with a distinct color.
box_trace_upcoming = create_box(x, y, z, dx, dy, dz, color='grey', opacity=0)
fig.add_trace(box_trace_upcoming, row=1, col=2)
outline_trace_upcoming = create_box_outline(x, y, z, dx, dy, dz, color='black', width=3)
fig.add_trace(outline_trace_upcoming, row=1, col=2)
# Optionally, add bin lines if you want a boundary around this separate coordinate system.
# add_bin_lines(fig, scene=2, W=W, D=D, H=H)

# Define a common camera view that we can apply to both scenes.
scene_camera = dict(
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1, y=-2.2, z=1.25)
)

# Update layout for both scenes.
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[0, W], ticks='outside', tickfont_size=15, title=dict(text='x', font=dict(size=25))),
        yaxis=dict(range=[D, 0], ticks='outside', tickfont_size=15, title=dict(text='y', font=dict(size=25))),
        zaxis=dict(range=[0, H], ticks='outside', tickfont_size=15, title=dict(text='z', font=dict(size=25))),
        camera=scene_camera,
        annotations=[
            dict(x=6, y=0, z=0.04, text="●",  # Unicode circle w/fill
                 showarrow=False, font=dict(size=20, color="red")),
            dict(x=6, y=0, z=0.04, text="(x=6,y=4)", showarrow=True, arrowwidth=2,
                 ax=0, ay=100, arrowcolor="black", font=dict(size=20, color="red"))
        ],
    ),
    scene2=dict(
        xaxis=dict(range=[0, W], title='', showbackground=False, showgrid=False, showticklabels=False),
        yaxis=dict(range=[D, 0], title='', showbackground=False, showgrid=False, showticklabels=False),
        zaxis=dict(range=[0, H], title='', showbackground=False, showgrid=False, showticklabels=False),
        bgcolor='rgba(0,0,0,0)',
        camera=scene_camera,
        annotations=[
            # dict(x=0, y=0, z=0, text="Upcoming item",  # Unicode circle w/fill
            #          showarrow=False, font=dict(size=20, color="red")),
            dict(x=0, y=4, z=0.04, text="●",  # Unicode circle w/fill
                 showarrow=False, font=dict(size=20, color="red")),
            # dict(x=6, y=4, z=0.04, text="(x=6,y=4)", showarrow=True, arrowwidth=2,
            #      ax=0, ay=100, arrowcolor="black", font=dict(size=20, color="red")),
        ],
    ),
    margin=dict(l=0, r=0, t=20, b=0),
    font_family='Computer Modern',
)

# fig.write_image('test.pdf')
# fig.update_layout(scene=dict(
#     annotations=[
#         dict(x=6, y=4, z=0.04, text="●",  # Unicode circle w/fill
#              showarrow=False, font=dict(size=20, color="red")),
#         dict(x=6, y=4, z=0.04, text="(x=6,y=4)", showarrow=True, arrowwidth=2,
#              ax=0, ay=100, arrowcolor="black", font=dict(size=20, color="red"))
#     ],
# ))

# pio.write_image(fig, FILENAME, scale=1, width=1200, height=1000)
fig.show()


