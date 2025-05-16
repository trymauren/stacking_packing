import numpy as np


class Item():

    def __init__(self, dimensions, mass):
        self.dimensions = len(dimensions)
        if self.dimensions == 2:
            self.len_x, self.len_y = dimensions
            self.len_x = int(self.len_x)
            self.len_y = int(self.len_y)
            self.x = -1
            self.y = -1

        else:
            self.len_x, self.len_y, self.len_z = dimensions
            self.len_x = int(self.len_x)
            self.len_y = int(self.len_y)
            self.len_z = int(self.len_z)
            self.x = -1
            self.y = -1
            self.z = -1

        self.mass = mass

    def get_position(self):
        if self.dimensions == 2:
            return np.asarray((self.x, self.y))
        else:
            return np.asarray((self.x, self.y, self.z))

    def get_position_mid(self):
        if self.dimensions == 2:
            return np.asarray((self.x + self.len_x/2,
                               self.y + self.len_y/2))
        else:
            return np.asarray((self.x + self.len_x/2,
                               self.y + self.len_y/2,
                               self.z + self.len_z/2))

    def get_dimensions(self):
        if self.dimensions == 2:
            return np.asarray((self.len_x, self.len_y))
        else:
            return np.asarray((self.len_x, self.len_y, self.len_z))

    def get_half_extents(self):
        return self.get_dimensions()/2

    def get_top_down_view(self, pallet_size):
        if self.dimensions == 2:
            return np.asarray([self.len_y for _ in pallet_size[0]])
        else:
            return np.asarray(
                [[self.len_z for _ in pallet_size[1]] for _ in pallet_size[0]])


if __name__ == '__main__':
    i = Item((1, 2, 3), 1)
    print(i.get_top_down_view())
