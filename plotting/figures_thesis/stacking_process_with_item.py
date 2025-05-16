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
golden = 1

def add_cube(ax, x, y, z, dx, dy, dz, color='blue', alpha=1.0):
    ax.bar3d(x, y, z, dx, dy, dz, color=color, alpha=alpha, shade=True)


def add_marker(ax, x, y, z, color='red', alpha=1):
    ax.scatter3D(x, y, z, color=color, alpha=alpha, label='FLB')


W, D, H = 10, 10, 10
# Create a figure and a 3D axes
fig = plt.figure(figsize=(fig_width, fig_width/golden))
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
ax5 = fig.add_subplot(2, 3, 5, projection='3d')
ax6 = fig.add_subplot(2, 3, 6)

for ax in [ax1, ax2, ax4, ax5]:
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)

# Optional: make the aspect ratio equal so cubes don’t look distorted
for ax in [ax1, ax2, ax4, ax5]:
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(D, 0)
    ax.set_zlim(0, H)

ax1.grid(False)
ax4.grid(False)
ax2.set_axis_off()
# ax3.set_axis_off()
ax5.set_axis_off()
# ax6.set_axis_off()

for ax in [ax1, ax4]:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    for x in range(W + 1):
        ax.plot([x, x], [0, D], [0, 0], color='gray', alpha=0.6)
    for y in range(D + 1):
        ax.plot([0, W], [y, y], [0, 0], color='gray', alpha=0.6)

    add_cube(ax, 0, 0, 0, 3, 3, 3, color='grey')
    add_cube(ax, 5, 0, 0, 2, 4, 2, color='grey')

    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_zticks([0, 2, 4, 6, 8])
    ax.tick_params(pad=-2)

add_cube(ax2, 0, 0, 0, 4, 4, 2, color='blue', alpha=0.3)
add_marker(ax2, 4, 4, 0)
ax2.legend()

add_cube(ax4, 0, 6, 0, 4, 4, 2, color='grey', alpha=0.3)
add_marker(ax4, 0, 6, 0)

add_cube(ax5, 0, 0, 0, 2, 2, 4, color='blue', alpha=0.3)
add_marker(ax5, 2, 2, 0)
ax5.legend()


A = np.array([
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
B = np.array([
    [3, 3, 3, 0, 0, 2, 2, 0, 0, 0],
    [3, 3, 3, 0, 0, 2, 2, 0, 0, 0],
    [3, 3, 3, 0, 0, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
])

for ax, data in zip([ax3, ax6], [A, B]):
    ax.set_box_aspect(1)
    ax.matshow(data, origin="upper", cmap="viridis", aspect="auto")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(data[i, j]), va='center', ha='center', size=6)
    ax.set_xlabel('x', labelpad=10)
    ax.set_ylabel('y', labelpad=10)
    ax.yaxis.set_label_position("right")
    ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    ax.tick_params(left=False, labelleft=False, right=True, labelright=True)
    ax.tick_params(axis='both', which='major', labelsize=6)
    # ax.tick_params(axis='x', which='major', labelleft=True, left=True)
    # ax.tick_params(axis='y', which='major', labelbottom=True, bottom=True)
    # ax.tick_params(axis='both', which='minor', labelsize=6)

ax1.set_title('Pallet state $t=2$', pad=20, fontsize=12)
ax2.set_title('Item to pack $t=2$', pad=20, fontsize=12)
ax3.set_title('Height map $t=2$', pad=20, fontsize=12)

ax4.set_title('Pallet state $t=3$', pad=20, fontsize=12)
ax5.set_title('Item to pack $t=3$', pad=20, fontsize=12)
ax6.set_title('Height map $t=3$', pad=20, fontsize=12)
# plt.tight_layout()

plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig('stacking_process_with_item.pdf', backend='pgf')
