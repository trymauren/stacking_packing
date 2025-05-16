import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import git
import sys

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root + '/code')
from plotting.figures_thesis.utils import *
plt.style.use(path_to_root + "/plot_config.mplstyle")


fig_width = 6*0.7
fig_height = 6*0.7


W, D, H = 10, 10, 10
fig = plt.figure(figsize=(fig_width, fig_height))
fig_row1, fig_row2 = fig.subfigures(nrows=2, ncols=1)

ax1 = fig_row1.add_subplot(1, 2, 1, projection='3d')
ax2 = fig_row1.add_subplot(1, 2, 2)
ax3 = fig_row2.add_subplot(1, 2, 1, projection='3d')
ax4 = fig_row2.add_subplot(1, 2, 2)

for ax in (ax1, ax3):
    setup_pallet(ax, W, D, H)
    add_cube(ax, 0, 0, 0, 3, 3, 3, color='grey', alpha=0.9)
    add_cube(ax, 5, 0, 0, 2, 4, 2, color='grey', alpha=0.9)
# add_cube(ax3, 4, 4, 0, 5, 4, 3, color='darkturquoise', alpha=1)
add_cube(ax3, 3, 0, 2, 5, 4, 3, color='grey', alpha=0.9)

height_map_1 = np.array([
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
height_map_2 = np.array([
    [3, 3, 3, 5, 5, 5, 5, 5, 0, 0],
    [3, 3, 3, 5, 5, 5, 5, 5, 0, 0],
    [3, 3, 3, 5, 5, 5, 5, 5, 0, 0],
    [0, 0, 0, 5, 5, 5, 5, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

for ax, height_map in zip((ax2, ax4), (height_map_1, height_map_2)):
    setup_and_plot_height_map(ax, height_map, vmax=H)

fig_row1.suptitle('Pallet state $S_2$', fontsize=12)
fig_row2.suptitle('Pallet state $S_3$', fontsize=12)
ax2.set_title('$\mathbf{M}_{2}$', fontsize=12)
ax4.set_title('$\mathbf{M}_{3}$', fontsize=12)
plt.subplots_adjust(wspace=0.7)
plt.savefig('pallet_state_transition.pdf', backend='pgf')