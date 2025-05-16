import numpy as np
import plotly.graph_objects as go

def generate_random_color(rng):
    return "#{:06x}".format(rng.integers(0, 0xFFFFFF))


def create_box(x, y, z, dx, dy, dz, **mesh_kwargs):
    # x, y = y, x
    """
    Creates a 3D box (rectangular prism) as a Plotly Mesh3d trace.
    (x, y, z) is the coordinate of one corner.
    (dx, dy, dz) are the dimensions of the box.
    """
    # Define the 8 vertices of the box
    vertices = [
        [x, y, z],             # vertex 0
        [x+dx, y, z],          # vertex 1
        [x+dx, y+dy, z],       # vertex 2
        [x, y+dy, z],          # vertex 3
        [x, y, z+dz],          # vertex 4
        [x+dx, y, z+dz],       # vertex 5
        [x+dx, y+dy, z+dz],    # vertex 6
        [x, y+dy, z+dz]        # vertex 7
    ]

    # Define the 12 triangles (two per face) using vertex indices.
    triangles = [
        (0, 1, 2), (0, 2, 3),   # bottom face
        (4, 5, 6), (4, 6, 7),   # top face
        (0, 1, 5), (0, 5, 4),   # front face
        (1, 2, 6), (1, 6, 5),   # right face
        (2, 3, 7), (2, 7, 6),   # back face
        (3, 0, 4), (3, 4, 7)    # left face
    ]
    i = [t[0] for t in triangles]
    j = [t[1] for t in triangles]
    k = [t[2] for t in triangles]

    return go.Mesh3d(
        x=[v[0] for v in vertices],
        y=[v[1] for v in vertices],
        z=[v[2] for v in vertices],
        i=i, j=j, k=k,
        flatshading=True,
        showlegend=False,
        **mesh_kwargs,
    )


def create_box_outline(x, y, z, dx, dy, dz, **line_kwargs):
    # y, x = x, y
    """
    Creates a Scatter3d trace with line segments along the edges of the box.
    """
    # Define the 8 vertices of the box (same as before)
    vertices = [
        [x, y, z],             # vertex 0
        [x+dx, y, z],          # vertex 1
        [x+dx, y+dy, z],       # vertex 2
        [x, y+dy, z],          # vertex 3
        [x, y, z+dz],          # vertex 4
        [x+dx, y, z+dz],       # vertex 5
        [x+dx, y+dy, z+dz],    # vertex 6
        [x, y+dy, z+dz]        # vertex 7
    ]

    # Define edges as pairs of vertex indices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face edges
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face edges
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]

    # Collect the x, y, z coordinates for each line segment, inserting None for breaks.
    x_lines, y_lines, z_lines = [], [], []
    for edge in edges:
        # Start vertex
        start_idx, end_idx = edge
        x_lines.extend([vertices[start_idx][0], vertices[end_idx][0], None])
        y_lines.extend([vertices[start_idx][1], vertices[end_idx][1], None])
        z_lines.extend([vertices[start_idx][2], vertices[end_idx][2], None])

    return go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        line=dict(**line_kwargs),
        showlegend=False,
    )


def add_bin_lines(fig, W, D, H, color='black', **trace_kwargs):

    fig.add_trace(go.Scatter3d(
        x=[W, W, 0],
        y=[0, D, D],
        z=[H, H, H],
        mode='lines',
        line=dict(color=color, width=3, dash='solid'),
        showlegend=False
    ), **trace_kwargs)

    fig.add_trace(go.Scatter3d(
        x=[W, 0, 0],
        y=[0, 0, D],
        z=[0, 0, 0],
        mode='lines',
        line=dict(color=color, width=3, dash='solid'),
        showlegend=False
    ), **trace_kwargs)

    fig.add_trace(go.Scatter3d(
        x=[W, W],
        y=[D, D],
        z=[0, H],
        mode='lines',
        line=dict(color=color, width=3, dash='solid'),
        showlegend=False
    ), **trace_kwargs)
    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, H],
        mode='lines',
        line=dict(color=color, width=3, dash='solid'),
        showlegend=False
    ), **trace_kwargs)

    fig.add_trace(go.Scatter3d(
        x=[W, W, 0],
        y=[0, D, D],
        z=[0, 0, 0],
        mode='lines',
        line=dict(color=color, width=3, dash='solid'),
        showlegend=False
    ), **trace_kwargs)
    fig.add_trace(go.Scatter3d(
        x=[W, 0, 0],
        y=[0, 0, D],
        z=[H, H, H],
        mode='lines',
        line=dict(color=color, width=3, dash='solid'),
        showlegend=False
    ), **trace_kwargs)

    fig.add_trace(go.Scatter3d(
        x=[W, W],
        y=[0, 0],
        z=[0, H],
        mode='lines',
        line=dict(color=color, width=3, dash='solid'),
        showlegend=False
    ), **trace_kwargs)
    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[D, D],
        z=[0, H],
        mode='lines',
        line=dict(color=color, width=3, dash='solid'),
        showlegend=False
    ), **trace_kwargs)
    return fig


def visual_setup(
    fig,
    W,
    D,
    H,
    title='',
    showgrid=False,
    showticks=False,
    x_title='',
    y_title='',
    z_title='',
):
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text=x_title, font=dict(size=25)),
                showticklabels=showticks,
                ticks='outside',
                tickfont_size=15,
                showgrid=showgrid,
                minallowed=0,
                maxallowed=W,
            ),
            yaxis=dict(
                title=dict(text=y_title, font=dict(size=25)),
                showticklabels=showticks,
                ticks='outside',
                tickfont_size=15,
                showgrid=showgrid,
                minallowed=0,
                maxallowed=D,
                range=(D, 0),
            ),
            zaxis=dict(
                title=dict(text=z_title, font=dict(size=25)),
                showticklabels=showticks,
                ticks='outside',
                tickfont_size=15,
                showgrid=showgrid,
                minallowed=0,
                maxallowed=H,
            ),
        ),
        scene_camera=dict(
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1, y=-2.2, z=1.25)
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        # title=dict(text=title, font=dict(size=20)),
        # showlegend=False,
        font_family='Computer Modern',
        # paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def visual_setup_2(
    fig,
    W,
    D,
    H,
    title='',
    showgrid=False,
    showticks=False,
    x_title='',
    y_title='',
    z_title='',
):
    fig.update_layout(
        scene2=dict(
            xaxis=dict(
                title=dict(text=x_title, font=dict(size=25)),
                showticklabels=showticks,
                ticks='outside',
                tickfont_size=15,
                showgrid=showgrid,
                minallowed=0,
                maxallowed=W,
            ),
            yaxis=dict(
                title=dict(text=y_title, font=dict(size=25)),
                showticklabels=showticks,
                ticks='outside',
                tickfont_size=15,
                showgrid=showgrid,
                minallowed=0,
                maxallowed=D,
                range=(D, 0),
            ),
            zaxis=dict(
                title=dict(text=z_title, font=dict(size=25)),
                showticklabels=showticks,
                ticks='outside',
                tickfont_size=15,
                showgrid=showgrid,
                minallowed=0,
                maxallowed=H,
            ),
        ),
        scene2_camera=dict(
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1, y=-2.2, z=1.25)
        ),
        # margin=dict(l=0, r=0, t=0, b=0),
        # title=dict(text=title, font=dict(size=20)),
        # showlegend=False,
        font_family='Computer Modern',
        # paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig