import numpy as np
import matplotlib.pyplot as plt
# m = i = y
# n = j = x


class Stack():

    def __init__(
            self,
            size,
            max_height,
            verbose: bool = True,
            ):

        self.y_lim, self.x_lim = size

        self.voxel_map = np.zeros(shape=(*size, max_height), dtype=int)  # CHANGE 30
        self.height_map = np.zeros(shape=size, dtype=int)
        self.max_height = max_height
        self.items = []
        self.verbose = verbose
        self._last_item_placed = 0
        self.total_volume_stacked = 0
        self.wasted_volume = 0

    def place(self, item):
        # print('hm: ', self.height_map)
        # print('pos: ', item.get_position())
        # print('dim: ', item.get_dimensions())

        x, y, z = item.get_position()
        len_x, len_y, len_z = item.get_dimensions()

        if x + len_x > self.x_lim:
            if self.verbose:
                self.print_out_of_bounds((x, y), (len_x, len_y))
            return 0

        if y + len_y > self.y_lim:
            if self.verbose:
                self.print_out_of_bounds((x, y), (len_x, len_y))
            return 0

        footprint = self.height_map[y:y + len_y, x:x + len_x]
        highest = np.amax(footprint)
        self.wasted_volume = np.sum(highest - footprint)

        item.z = z = highest

        if z + len_z >= self.max_height:
            if self.verbose:
                self.print_too_high(z, len_z)
            return 0

        self.height_map[y:y + len_y, x:x + len_x] = z + len_z

        self.voxel_map[y:(y + len_y),
                       x:(x + len_x),
                       z:(z + len_z)] = 1

        self.total_volume_stacked += (len_x*len_y*len_z)
        self.items.append(item)
        self._last_item_placed = item

        return 1

    def add_z_pos_to_item(self, item):
        x, y, _ = item.get_position()
        len_x, len_y, _ = item.get_dimensions()
        item.z = np.amax(self.height_map[y:y + len_y, x:x + len_x])

    def height(self):
        return np.max(self.height_map)

    def get_wasted_space_ratio(self):
        return self.wasted_volume / (self.x_lim * self.y_lim)

    def get_footprint_support_ratio(self, item):
        x, y, _ = item.get_position()
        len_x, len_y, _ = item.get_dimensions()

        footprint = self.height_map[y:y + len_y, x:x + len_x]
        highest = np.amax(footprint)
        support_ratio = (
            np.sum(highest == footprint) / footprint.size
        )
        return support_ratio

    def get_voxel_map(self):
        return self.voxel_map

    def get_height_map(self):
        return self.height_map

    def get_stacked_items(self):
        return self.items

    def get_dimensions(self):
        return self.x_lim, self.y_lim, self.max_height

    def observe(self):
        return self.height_map

    def print_me(self):
        print(self.height_map)

    def print_out_of_bounds(self, cor, dim):
        out = f'Prevented stacking of item at position'
        out += f' x:{cor[0]}->{cor[0]+dim[0]}, y:{cor[1]}->{cor[1]+dim[1]},'
        out += f' because out of bounds.'
        print(out)

    def print_too_high(self, z, len_z):
        out = f'Prevented stacking of item, because it would exceed max '
        out += f'height of pallet.'
        print(out)

    def plot_stack(self):

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        _x = range(len(self.height_map[0]))
        _y = range(len(self.height_map))
        xx, yy = np.meshgrid(_x, _y)
        x, y = xx.ravel(), yy.ravel()
        z = np.zeros_like(x)
        width_of_bar = np.ones_like(x)
        depth_of_bar = np.ones_like(x)
        height_of_bar = self.height_map.ravel()
        print(height_of_bar)
        ax.bar3d(x, y, z,
                 width_of_bar,
                 depth_of_bar,
                 height_of_bar,
                 shade=True
                 )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def optimal_height(self, width_of_strip):  # not done
        highest_item = max(self.items, key=attrgetter('len_z')).height
        items_h = sum(item.x * item.y * item.z for item in self.items)/width_of_strip
        return min(highest_item, items_h)

    def worst_height(self):  # not done
        height = 0
        for item in self.items:
            height += item.height
        return height

    def last_item_placed(self):
        return self._last_item_placed

    def is_out_of_bounds(self, item):
        x, y, _ = item.get_position()
        len_x, len_y, len_z = item.get_dimensions()

        if x + len_x > self.x_lim:
            if self.verbose:
                self.print_out_of_bounds((x, y), (len_x, len_y))
            return 0

        if y + len_y > self.y_lim:
            if self.verbose:
                self.print_out_of_bounds((x, y), (len_x, len_y))
            return 0

        highest = np.amax(self.height_map[y:(y + len_y), x:(x + len_x)])

        z = highest

        if z + len_z >= self.max_height:
            if self.verbose:
                self.print_too_high(z, len_z)
            return 0
        return 1

    def get_corner_support_index(self, item):

        x, y, _ = item.get_position()
        len_x, len_y, len_z = item.get_dimensions()

        # if x + len_x > self.x_lim:
        #     raise IndexError('Attempted to place item outside self')

        # if y + len_y > self.y_lim:
        #     raise IndexError('Attempted to place item outside self')

        def get_z(x, y):
            return self.height_map[y, x]

        flb = x, y, get_z(x, y)  # front-left-bottom
        rlb = x, y + len_y-1, get_z(x, y + len_y-1)  # rear-left-bottom
        frb = x + len_x-1, y, get_z(x + len_x-1, y)  # front-right-bottom
        rrb = x + len_x-1, y + len_y-1, get_z(x + len_x-1, y + len_y-1)  # rear-right-bottom

        highest = np.amax(self.height_map[
            y:(y + len_y),
            x:(x + len_x)]
            )

        corners = flb, rlb, frb, rrb

        corners_supported = 0

        for corner in corners:
            corners_supported += int(corner[-1] == highest)

        return corners_supported

    def get_area_support_score(self, item):

        x, y, _ = item.get_position()
        len_x, len_y, len_z = item.get_dimensions()

        # if x + len_x > self.x_lim:
        #     raise IndexError('Attempted to place item outside self')

        # if y + len_y > self.y_lim:
        #     raise IndexError('Attempted to place item outside self')

        # def get_z(x, y):
        #     return self.height_map[y, x]

        # flb = x, y, get_z(x, y)  # front-left-bottom
        # rlb = x, y + len_y-1, get_z(x, y + len_y-1)  # rear-left-bottom
        # frb = x + len_x-1, y, get_z(x + len_x-1, y)  # front-right-bottom
        # rrb = x + len_x-1, y + len_y-1, get_z(x + len_x-1, y + len_y-1)  # rear-right-bottom

        highest = np.amax(self.height_map[y:(y + len_y), x:(x + len_x)])
        same = (self.height_map[y:(y + len_y), x:(x + len_x)] == highest)

        return np.mean(same)

    def placement_feasible(coordinate, item_dimensions, self):

        x, y = coordinate
        len_x, len_y, len_z = item_dimensions

        if x + len_x > self.x_lim:
            return 0

        if y + len_y > self.y_lim:
            return 0

        highest = np.amax(self.height_map[
            y:(y + len_y),
            x:(x + len_x)]
            )

        z = highest

        if z + len_z > self.max_height:
            return 0

        return 1

    def get_ob_mask(self, item_shape, flat_action_space):

        len_x, len_y, len_z = item_shape
        max_y, max_x = self.height_map.shape
        max_z = self.max_height

        max_col_pos = max_x - len_x
        max_row_pos = max_y - len_y
        max_z_pos = max_z - len_z

        if flat_action_space:
            ob_mask = np.zeros_like(self.height_map, dtype=np.bool_)
            ob_mask[:max_row_pos + 1, :max_col_pos + 1] = 1
            mask = ob_mask  # & (self.height_map <= max_z)
            return mask
        else:
            x_mask = np.full(shape=(max_x), fill_value=False, dtype=bool)
            y_mask = np.full(shape=(max_y), fill_value=False, dtype=bool)
            x_mask[:max_col_pos + 1] = True
            y_mask[:max_row_pos + 1] = True
            mask = np.concatenate((y_mask, x_mask), axis=0)
            return mask

    def get_support_mask(self, item_shape, min_support):
        len_x, len_y, len_z = item_shape
        max_y, max_x = self.height_map.shape
        max_col_pos = max_x - len_x
        max_row_pos = max_y - len_y

        area_support_mask = np.zeros_like(self.height_map, dtype=np.bool_)

        for row in range(max_row_pos + 1):
            for col in range(max_col_pos + 1):
                slice_ = self.height_map[row:row+len_y, col:col+len_x]
                max_height = np.amax(slice_)
                same_as_max_height = (slice_ >= max_height).sum()
                support_ratio = same_as_max_height / slice_.size
                ix_supported = support_ratio > min_support
                area_support_mask[row, col] = ix_supported

                if not ix_supported:  # can still have edges supported enough
                    pass  # TODO: implement edge-support detection

        return area_support_mask

    def valid_placements(
        self,
        item_shape,
        flat_action_space=True,
        min_support=0
    ):
        ob_mask = self.get_ob_mask(item_shape, flat_action_space)
        if min_support:
            support_mask = self.get_support_mask(
                item_shape=item_shape,
                min_support=min_support
            )
            mask = ob_mask & support_mask
        else:
            mask = ob_mask

        return mask

    def gap_ratio(self):
        if self.height() == 0:
            return 0
        pallet_volume = self.x_lim * self.y_lim * self.height()
        return 1 - np.divide(self.total_volume_stacked, pallet_volume)

    def solving_packing_problems_gap_ratio(self):
        pallet_volume = self.x_lim * self.y_lim * self.height()
        return pallet_volume - self.total_volume_stacked

    def compactness(self):
        if self.height() == 0:
            return 0
        pallet_volume = self.x_lim * self.y_lim * self.height()
        return np.divide(self.total_volume_stacked, pallet_volume)

    def place_flb_heuristic(self, item):

        height_map = self.height_map

        found_x = 0
        found_y = 0
        found_z = 0

        y_x_len = np.asarray([item.len_y, item.len_x])
        it = np.nditer(height_map, flags=['multi_index'])
        for i in it:
            slice_ = i + y_x_len
            hei = height_map[slice_]
            print(hei)

        item.x = found_x
        item.y = found_y
        item.z = found_z

        self.place(item)
