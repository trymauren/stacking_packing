import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # This import is needed for 3D plotting
import git
import sys

path_to_root = git.Repo('.', search_parent_directories=True).working_dir

# matplotlib.use("pgf")
# Configure the LaTeX-related settings
plt.style.use(path_to_root + "/plot_config.mplstyle")
# matplotlib.rcParams.update({
#     "text.usetex": True,
#     "pgf.texsystem": "pdflatex",
#     "pgf.rcfonts": False,
#     "font.family": "serif",
#     "font.size": 12,
#     # "text.latex.preamble": r"\usepackage{amsmath}",
# })

fig_width = 6
# golden = (1 + 5 ** 0.5) / 2


def add_cube(ax, x, y, z, dx, dy, dz, color='blue', alpha=1.0):
    ax.bar3d(x, y, z, dx, dy, dz, color=color, alpha=alpha, shade=True)


def add_marker(ax, x, y, z, color='red', alpha=1):
    ax.scatter3D(x, y, z, color=color, alpha=alpha, label='FLB')


W, D, H = 10, 10, 10
# Create a figure and a 3D axes
fig = plt.figure(figsize=(fig_width, 7))
subfigs = fig.subfigures(nrows=2, ncols=1)
subfigs[0].suptitle('State at $t=2$')
subfigs[1].suptitle('State representation at $t=2$')

ax1 = subfigs[0].add_subplot(1, 2, 1, projection='3d')
ax3 = subfigs[0].add_subplot(1, 2, 2, projection='3d')
ax2 = subfigs[1].add_subplot(1, 2, 1)
ax4 = subfigs[1].add_subplot(1, 2, 2)

ax2.set_axis_off()
ax4.set_axis_off()

for ix, ax in enumerate([ax1, ax3]):
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)

    ax.set_box_aspect((1, 1, 1))

    ax.grid(False)

    ax.set_xlabel('x', labelpad=-2.5)
    ax.set_ylabel('y', labelpad=-2.5)
    ax.set_zlabel('z', labelpad=-2.5)

    if ix == 0:
        ax.set_xlim(0, W)
        ax.set_ylim(D, 0)
        ax.set_zlim(0, H)
        for x in range(W + 1):
            ax.plot([x, x], [0, D], [0, 0], color='gray', alpha=0.6)
        for y in range(D + 1):
            ax.plot([0, W], [y, y], [0, 0], color='gray', alpha=0.6)

        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_yticks([0, 2, 4, 6, 8])
        ax.set_zticks([0, 2, 4, 6, 8])
        ax.tick_params(pad=-2)

    if ix == 1:
        ax.set_xlim(0, 4)
        ax.set_ylim(4, 0)
        ax.set_zlim(0, 4)
        for x in range(4 + 1):
            ax.plot([x, x], [0, 4], [0, 0], color='gray', alpha=0.6)
        for y in range(4 + 1):
            ax.plot([0, 4], [y, y], [0, 0], color='gray', alpha=0.6)

        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 1, 2, 3])
        ax.set_zticks([0, 1, 2, 3])
        ax.tick_params(pad=-2)

add_cube(ax1, 0, 0, 0, 3, 3, 3, color='grey')
add_cube(ax1, 5, 0, 0, 2, 4, 2, color='grey')

add_cube(ax3, 0, 0, 0, 4, 4, 2, color='grey')

M = np.array([
    [3, 3, 3, 0, 0, 2, 2, 0, 0, 0],
    [3, 3, 3, 0, 0, 2, 2, 0, 0, 0],
    [3, 3, 3, 0, 0, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

ax2.set_box_aspect(1)
ax4.set_box_aspect(1)

ax2.text(0.5, 1, M, fontsize=12, ha='center', va='top', transform=ax2.transAxes)
ax4.text(0.5, 1, '$[[4, 4, 2, 10]]$', fontsize=12, ha='center', va='top', transform=ax4.transAxes)

ax1.set_title('Pallet', fontsize=12)
ax2.set_title('$\mathbf{M}_{2}$', fontsize=12)
ax3.set_title('Upcoming item (k=0)', fontsize=12)
ax4.set_title('$\mathbf{i}_{2}$', fontsize=12)
# subfigs[0].align_titles([ax1, ax3])
# subfigs[1].align_titles([ax2, ax4])
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig('state_representation_matrix.pdf', backend='pgf')
