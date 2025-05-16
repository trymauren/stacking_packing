import sys
import git
import matplotlib.pyplot as plt
import numpy as np
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root + '/code')
from plotting.plt_utils import *
from stacking_objects.item import Item
# plt.style.use(path_to_root + "/plot_config.mplstyle")
SET1_CMAP = plt.cm.get_cmap('Set1')


def plt_plot_stack(
    items,
    W,
    D,
    show=False,
    save_path=False,
    rng=829,
    fig_width=6*0.7,
    fig_height=6*0.7
):
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    fig_width = 6*0.7
    fig_height = 6*0.7

    H = max(W, D)
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    setup_pallet(
        ax1, W, D, H, labelsize=6, labelpad=-7, tickpad=-5,
        grid_color='grey', grid_alpha=1
    )

    for i, item in enumerate(items):
        dims = item.get_dimensions()
        pos = item.get_position()
        add_cube(
            ax1, *pos, *dims, color=SET1_CMAP(i % 9), alpha=1,
        )

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)
