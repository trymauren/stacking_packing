import numpy as np
import matplotlib.pyplot as plt
import pygame

class Item(object):

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def rotate(self):
        pass


class Strip(object):

    def __init__(self, width: int, height: int, game=False):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))
        self.contour = np.zeros(int(width))
        self.clock = 0
        if game:
            pygame.init()
            pygame.display.set_caption('Packer 1')
            self.clock = pygame.time.Clock()
            self.screen = pygame.display.set_mode((width, height))

    def place(self, item):  # the fun stuff
        c = np.inf
        idx_c = []
        idx_temp = []
        temp = 0
        for i in range(len(self.contour)):
            if self.contour[i] == temp:
                idx_temp.append(i)
            else:
                idx_temp = []
                idx_temp.append(i)
                temp = self.contour[i]

            if len(idx_temp) == item.width:
                if temp < c:
                    c = self.contour[i]
                    idx_c = idx_temp[:]
                idx_temp = []  # unecessary?
                temp = np.inf

        if len(idx_c) == item.width:
            for i in idx_c:
                for j in range(int(c), int(c + item.height)):
                    self.grid[i][j] = 1
                self.contour[i] += item.height
        else:
            print('Must place non-optimal')

            for i in range(item.width):
                for j in range(int(self.contour[0]), int(self.contour[0] + item.height)):
                    self.grid[i][j] = 1

            for i in range(item.width):
                self.contour[i] += item.height

        if self.clock:
            self.update_display()

    def update_display(self):
        self.screen.fill((255, 255, 255))  # White background
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == 1:
                    pygame.draw.rect(
                                     self.screen,
                                     (0, 0, 0),
                                     (x, self.height - y - 1, 1, 1)
                                     )
        pygame.display.flip()

    def run(self, items):
        i = 0
        running = True
        while i < len(items):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.place(items[i])  # Example item placement
            self.clock.tick(1)  # Run at 60 FPS
            print(i)
            i += 1
            if i == len(items):
                i = 0
        pygame.quit()


def main():
    s = Strip(500, 500, game=True)
    item1 = Item(50, 50)
    item2 = Item(100, 20)
    item3 = Item(70, 10)
    item4 = Item(10, 100)
    item5 = Item(100, 100)
    item6 = Item(10, 100)
    items = [item1, item2, item3, item4, item5, item6]
    # items = [Item(5, 5, 0) for _ in range(40)]
    s.run(items)


main()
