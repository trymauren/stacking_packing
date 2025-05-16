import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # This import is needed for 3D plotting
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
ax2 = fig.add_subplot(1, 2, 2)

# plot_one_item_in_grid(ax1, 0, 0, 0, 4, 4, 2, H=4)
plot_one_item_in_space(ax1, 0, 0, 0, 8, 8, 4, W, D, H, cube_color='grey', cube_alpha=0.9)
add_text_marker(ax1, 1, 12, -1, '$d=4$', color='black')
add_text_marker(ax1, 9.2, 5, 0, '$w=4$', color='black')
add_text_marker(ax1, 10.5, 4, 4, '$h=2$', color='black')
add_text_marker(ax1, 0, 5, 6.8, '$m=10$', color='black')

setup_and_plot_item_vectors(
    ax2, text='$[[4, 4, 2, 10]]$', fontsize=12,
    ha='center', va='top', transform=ax2.transAxes
)

ax1.set_title('Upcoming item state', pad=20, fontsize=12)
ax2.set_title('State representation $\mathbf{i}_t = $', pad=20, fontsize=12)

plt.subplots_adjust(wspace=1)
plt.savefig('upcoming_item.pdf', backend='pgf')
