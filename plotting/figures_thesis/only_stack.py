import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import git
import sys

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
from plotting.figures_thesis.utils import *

fig_width = 1.2
fig_height = 1.2


W, D, H = 10, 10, 10
fig = plt.figure(figsize=(fig_width, fig_height))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')

setup_minimal_pallet(ax1, W, D, H)
add_cube(ax1, 0, 0, 0, 4, 4, 7, color='grey')
add_cube(ax1, 6, 0, 0, 2, 4, 2, color='grey')
add_cube(ax1, 0, 6, 0, 4, 2, 4, color='grey')

plt.subplots_adjust(wspace=1)
plt.savefig('stack_for_drawio.pdf')