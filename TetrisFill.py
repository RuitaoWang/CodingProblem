"""
Tetris filling given rectangle problem of certain size
"""
import numpy as np
import itertools
from copy import deepcopy


class Tetris:

    def __init__(self, name, cor):
        self.name = name
        self.cor = np.array(cor)
        self.length, self.height = self.cor.max(0)

    def print(self):
        forprint = np.zeros([4, 4], dtype=np.int)
        for _ in self.cor:
            forprint[_[0], _[1]] = self.name
        forprint = np.flipud(np.transpose(forprint))

        print(forprint)


class Grid:

    def __init__(self, size):
        self.cor = np.zeros(size, dtype=np.int)
        self.length, self.height = size

    def print(self):
        print(np.flipud(np.transpose(self.cor)))

    def place_tetris(self, tetris, x, y):

        # check if placeable
        place_cor = tetris.cor + [x, y]
        place_cor_x_max, place_cor_y_max = place_cor.max(0)
        # boundary check
        if (place_cor_x_max >= grid.length) or (place_cor_y_max >= grid.height):
            return False, self
        # already exist

        if np.sum([self.cor[x[0], x[1]] for x in place_cor]) > 0:
            return False, self

        # place
        new_grid = deepcopy(self)
        for bit in place_cor:
            new_grid.cor[bit[0], bit[1]] = tetris.name
        return True, new_grid
    @property
    def if_filled(self):
        return np.all(self.cor)

name2cordinate_tetris = {
    'L': [[0,0], [0,1], [0,2], [1,2]],
    'L90': [[0,0], [1,0], [2,0], [0,1]],
    'L180': [[0,0], [1,0], [1,1], [1,2]],
    'L270': [[2,0], [0,1], [1,1], [2,1]],
    'T': [[0,0], [1,0], [2,0], [1,1]],
    'T90': [[0,0], [0,1], [0,2], [1,1]],
    'T180': [[0,1], [1,1], [2,1], [1,0]],
    'T270': [[0,1], [1,0], [1,1], [1,2]],
    'J': [[1,0], [1,1], [0,2], [1,2]],
    'J180': [[0,0], [1,0], [0,1], [0,2]],
    'Qtr': [[0,0], [1,0], [1,1]],
    'Qbl': [[0,0], [0,1], [1,1]],
    'S90': [[0,0], [0,1], [1,1], [1,2]],
    'plus': [[1,0], [0,1], [1,1], [2,1], [1,2]],
    '1x1': [[0,0]],
    'long':[[0,0],[1,0],[2,0],[3,0]]
}

def generate_single_tetris(tetris_name, tetris_type, tetris_dict):
    """
    """
    return Tetris(tetris_name, tetris_dict[tetris_type])


def solve_fill_in_problem(grid, tetris_lists):
    if (len(tetris_lists) == 0) or grid.if_filled:
        return grid
    else:
        for y in range(grid.height):
            for x in range(grid.length):
                placed, new_grid = grid.place_tetris(tetris_lists[0], x, y)
                if placed:
                    return solve_fill_in_problem(new_grid, tetris_lists[1:])
        solved = solve_fill_in_problem(new_grid, tetris_lists[1:])

        return solved


def solve(grid, tetris_lists):
    te_l = (list(x) for x in itertools.permutations(tetris_lists))

    for tl in te_l:
        res = solve_fill_in_problem(grid, tl)
        if res.if_filled:
            break
    return res


if __name__ == '__main__':
    grid = Grid((2, 2))
    block_l = ['L', 'L90', 'L180', 'L180', 'plus', 'T', 'L270', 'J', 'J180', 'Qtr',
               'Qtr', 'Qbl', 'S90']
    block_t = [generate_single_tetris(i + 1, block_l[i], name2cordinate_tetris) for i in range(len(block_l))]
    solve(grid, block_t).print()