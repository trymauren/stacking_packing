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
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

setup_pallet(ax1, W, D, H)

plot_one_item_in_space(ax2, 0, 0, 0, 5, 4, 3, W, D, H, cube_color='grey', cube_alpha=0.4)
add_text_marker(ax2, 3, 7, 0, '$d$', color='black')
add_text_marker(ax2, 5, 2.2, -1, '$w$', color='black')
add_text_marker(ax2, 6, 1.5, 1.5, '$h$', color='black')
add_marker(ax2, 5, 4, 0, color='red', alpha=1)
add_cube(ax1, 4, 4, 0, 5, 4, 3, color='grey', alpha=0.4)
add_marker(ax1, 4, 4, 0, color='red', alpha=1)

# fig.suptitle('$A_t = (4, 4)$. FLB is positioned at $x=4,y=4$', fontsize=11, y=0.75)
plt.legend()
plt.savefig('placement_action.pdf', backend='pgf')