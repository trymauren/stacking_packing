import numpy as np


class ItemLoader(object):
    def __init__(
        self,
        item_getter_function,
        i_g_f_kwargs,
        reset_at_step,
    ):
        self.item_getter_function = item_getter_function
        self.i_g_f_kwargs = i_g_f_kwargs
        self.reset_at_step = reset_at_step
        self.n_steps = 0

    def __call__(self):
        if self.n_steps == self.reset_at_step:
            self.reset()
        items = self.item_getter_function(rng=self.rng, **self.i_g_f_kwargs)
        self.n_steps += 1
        return items

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def reset(self):
        self.n_steps = 0
        self.rng = np.random.default_rng(self.seed)
