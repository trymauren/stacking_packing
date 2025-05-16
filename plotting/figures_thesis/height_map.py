import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import git
import sys

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root + '/code')
from plotting.figures_thesis.utils import *

fig_width = 6*0.7
fig_height = 6*0.7


W, D, H = 10, 10, 10
fig = plt.figure(figsize=(fig_width, fig_height))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2)

setup_pallet(ax1, W, D, H)
add_cube(ax1, 0, 0, 0, 3, 3, 3, color='grey')
add_cube(ax1, 5, 0, 0, 2, 4, 2, color='grey')

height_map = np.array([
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
setup_and_plot_height_map(ax2, height_map, vmax=H)

ax1.set_title('Pallet state', pad=20, fontsize=12)
ax2.set_title('State representation $\mathbf{H}_t = $', pad=20, fontsize=12)

plt.subplots_adjust(wspace=1)
plt.savefig('height_map.pdf')