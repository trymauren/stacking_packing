import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import git
import sys

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root + '/code')
from plotting.figures_thesis.utils import *

fig_width = 1
fig_height = 1


W, D, H = 10, 10, 10
fig = plt.figure(figsize=(fig_width, fig_height))
ax1 = fig.add_subplot(1, 1, 1)

height_map = np.array([
    [7, 7, 7, 7, 0, 0, 2, 2, 0, 0],
    [7, 7, 7, 7, 0, 0, 2, 2, 0, 0],
    [7, 7, 7, 7, 0, 0, 2, 2, 0, 0],
    [7, 7, 7, 7, 0, 0, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])
setup_and_plot_minimal_height_map(ax1, height_map, vmax=H)

plt.subplots_adjust(wspace=1)
plt.savefig('height_map_for_drawio.pdf')