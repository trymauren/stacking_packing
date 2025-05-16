import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import git
import sys

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
plt.style.use(path_to_root + '/thesis/plot_config.mplstyle')


def add_cube(ax, x, y, z, dx, dy, dz, color='blue', alpha=1.0):
    ax.bar3d(x, y, z, dx, dy, dz, color=color, alpha=alpha, shade=True)


def add_marker(ax, x, y, z, color='red', alpha=1, label='FLB'):
    ax.scatter3D(x, y, z, color=color, alpha=alpha, label=label)


def add_text_marker(ax, x, y, z, text, color='black', alpha=1, **text_kwargs):
    ax.text(x, y, z, text, color=color, alpha=alpha, **text_kwargs)


def setup_and_plot_height_map(ax, height_map, vmax=10):
    W, D = height_map.shape
    ax.set_box_aspect(1)
    ax.matshow(height_map, origin='upper', cmap='YlGnBu', vmin=0, vmax=vmax, aspect='auto')
    for i in range(W):
        for j in range(D):
            ax.text(j, i, str(height_map[i, j]), va='center', ha='center', size=6)
    ax.set_xlabel('$x$', labelpad=10)
    ax.set_ylabel('$y$', labelpad=10, rotation=0)
    ax.yaxis.set_label_position('right')
    ax.set_xticks(range(0, W, 2))
    ax.set_xticks(range(0, W, 1), minor=True)
    ax.set_yticks(range(0, D, 2))
    ax.set_yticks(range(0, D, 1), minor=True)
    ax.tick_params(
        axis='both', which='both',
        top=False, left=False, right=True, bottom=True,
        labeltop=False, labelleft=False, labelright=True, labelbottom=True,
        labelsize=6,
    )
    # ax.tick_params(axis='both', which='major', labelsize=6)


def setup_and_plot_minimal_height_map(ax, height_map, vmax=10, cmap='YlGnBu'):
    W, D = height_map.shape
    ax.set_box_aspect(1)
    ax.matshow(height_map, origin='upper', cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
    for i in range(W):
        for j in range(D):
            ax.text(j, i, str(height_map[i, j]), va='center', ha='center', size=4)
    ax.set_xlabel('$x$', labelpad=0)
    ax.set_ylabel('$y$', labelpad=0)
    ax.yaxis.set_label_position('right')
    ax.tick_params(
        axis='both', which='both',
        top=False, left=False, right=False, bottom=False,
        labeltop=False, labelleft=False, labelright=False, labelbottom=False,
        labelsize=6,
    )


def setup_and_plot_item_vectors(ax, text, **text_kwargs):
    ax.set_axis_off()
    ax.set_box_aspect(1)
    ax.text(0.5, 1, text, **text_kwargs)

    # ax.text(
    #     0.5, 0.5, '$\;\;\;\; = [[4, 4, 2, 10]]$', fontsize=20,
    #     ha='center', va='center', transform=ax.transAxes
    # )
    # ax.text(
    #     0.5, 0.8, '$\mathbf{i}_t = [[w \; d \; h \; m]]$', fontsize=20,
    #     ha='center', va='center', transform=ax4.transAxes
    # )


def setup_pallet(
    ax,
    W, D, H,
    labelsize=6, labelpad=-7, tickpad=-5,
    grid_color='grey', grid_alpha=1
):
    ax.tick_params(
        axis='both', which='major', labelsize=labelsize, pad=tickpad
    )
    ax.tick_params(
        axis='both', which='minor', labelsize=labelsize, pad=tickpad
    )
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('$x$', labelpad=labelpad)
    ax.set_ylabel('$y$', labelpad=labelpad)
    ax.set_zlabel('$z$', labelpad=labelpad)
    ax.set_xlim(0, W)
    ax.set_ylim(D, 0)
    ax.set_zlim(0, H)
    ax.set_xticks(range(1, W+1, 2))
    ax.set_xticks(range(0, W+1, 1), minor=True)
    ax.set_yticks(range(1, D+1, 2))
    ax.set_yticks(range(0, D+1, 1), minor=True)
    ax.set_zticks(range(1, H+1, 2))
    ax.set_zticks(range(0, H+1, 2), minor=True)
    ax.grid(which='both')


def setup_minimal_pallet(
    ax,
    W, D, H,
    labelsize=6, labelpad=-7, tickpad=-5,
    grid_color='grey', grid_alpha=1
):
    ax.tick_params(
        axis='both', which='major', labelsize=labelsize, pad=tickpad
    )
    ax.tick_params(
        axis='both', which='minor', labelsize=labelsize, pad=tickpad
    )
    ax.set_box_aspect((1, 1, 1))
    # ax.set_xlabel('$x$', labelpad=labelpad, fontsize=)
    # ax.set_ylabel('$y$', labelpad=labelpad, fontsize=)
    # ax.set_zlabel('$z$', labelpad=labelpad, fontsize=)
    ax.set_xlim(0, W)
    ax.set_ylim(D, 0)
    ax.set_zlim(0, H)
    ax.set_xticks(range(1, W+1, 2))
    ax.set_xticks(range(0, W+1, 1), minor=True)
    ax.set_yticks(range(1, D+1, 2))
    ax.set_yticks(range(0, D+1, 1), minor=True)
    ax.set_zticks(range(1, H+1, 2))
    ax.set_zticks(range(0, H+1, 2), minor=True)
    ax.grid(which='both')


def plot_one_item_in_grid(
    ax,
    x, y, z,
    dx, dy, dz,
    H,
    cube_color='grey', cube_alpha=1,
    labelsize=6, labelpad=-2.5, tickpad=-2,
    grid_color='grey', grid_alpha=1
):
    ax.tick_params(
        axis='both', which='major', labelsize=labelsize, pad=tickpad
    )
    ax.tick_params(
        axis='both', which='minor', labelsize=labelsize, pad=tickpad
    )
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('$x$', labelpad=labelpad)
    ax.set_ylabel('$y$', labelpad=labelpad)
    ax.set_zlabel('$z$', labelpad=labelpad)
    ax.set_xlim(0, dx)
    ax.set_ylim(dy, 0)
    ax.set_zlim(0, H)
    ax.set_xticks(range(0, dx+1, 1))
    ax.set_xticks(range(0, dx+1, 1), minor=True)
    ax.set_yticks(range(0, dy+1, 1))
    ax.set_yticks(range(0, dy+1, 1), minor=True)
    ax.set_zticks(range(0, dz+1, 1))
    ax.set_zticks(range(0, H+1, 1), minor=True)
    ax.grid(which='both')
    add_cube(ax, x, y, z, dx, dy, dz, color=cube_color, alpha=cube_alpha)


def plot_one_item_in_space(
    ax,
    x, y, z,
    dx, dy, dz,
    W, D, H,
    cube_color='grey', cube_alpha=1,
    labelsize=6, labelpad=-2.5, tickpad=-2,
    grid_color='grey', grid_alpha=1
):
    ax.set_axis_off()
    ax.tick_params(
        axis='both', which='major', labelsize=labelsize, pad=tickpad
    )
    ax.tick_params(
        axis='both', which='minor', labelsize=labelsize, pad=tickpad
    )
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(D, 0)
    ax.set_zlim(0, H)
    add_cube(ax, x, y, z, dx, dy, dz, color=cube_color, alpha=cube_alpha)
