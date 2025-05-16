import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # This import is needed for 3D plotting
import git
import sys

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
from plotting.figures_thesis.utils import *
plt.style.use(path_to_root + "/plot_config.mplstyle")

fig_width = 6
W, D, H = 10, 10, 10

fig = plt.figure(figsize=(fig_width, 7))
subfigs = fig.subfigures(nrows=2, ncols=1)
subfigs[0].suptitle('State at $t=2$')
subfigs[1].suptitle('State representation at $t=2$')

ax1 = subfigs[0].add_subplot(1, 2, 1, projection='3d')
ax3 = subfigs[0].add_subplot(1, 2, 2, projection='3d')
ax2 = subfigs[1].add_subplot(1, 2, 1)
ax4 = subfigs[1].add_subplot(1, 2, 2)

ax4.set_axis_off()

plot_one_item_in_grid(ax3, 0, 0, 0, 4, 4, 2, H=4, cube_color='grey', cube_alpha=1)

setup_pallet(ax1, W, D, H)
add_cube(ax1, 0, 0, 0, 3, 3, 3, color='grey')
add_cube(ax1, 5, 0, 0, 2, 4, 2, color='grey')

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

setup_and_plot_height_map(ax2, M, vmax=H)
setup_and_plot_item_vectors(
    ax4, text='$[[4, 4, 2, 10]]$', fontsize=12,
    ha='center', va='top', transform=ax4.transAxes
)

ax1.set_title('Pallet', fontsize=12)
ax2.set_title('$\mathbf{M}_{2}$', fontsize=12)
ax3.set_title('Upcoming item (k=0)', fontsize=12)
ax4.set_title('$\mathbf{i}_{2}$', fontsize=12)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig('state_representation.pdf', backend='pgf')
