import random
import numpy as np
from scipy.signal import convolve2d


def generate_mines(rows=9, col=9, mine_chance=0.125):
    global mine_count
    mine_count = 0

    board = np.zeros((rows, col), dtype='int32')
    startx = random.randint(0, rows - 1)
    starty = random.randint(0, col - 1)

    # Define neighbors for (startx, starty)
    neighbor_indices = [(startx + dx, starty + dy) for dx in range(-1, 2) for dy in range(-1, 2)]

    while mine_count < rows * col * mine_chance - 1:
        minex = random.randint(0, rows - 1)
        miney = random.randint(0, col - 1)
        # Check if the new mine is adjacent to startx or starty
        if (minex, miney) in neighbor_indices:
            continue
        if board[minex, miney] != 9:
            board[minex, miney] = 9
            mine_count += 1
    board = np.array(board, dtype='int32')
    return board, startx, starty


def add_one_to_nines(array):
    # Define the convolution kernel
    array = np.array(array, dtype='int32')
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # Convolve the input array with the kernel for regular bombs (9)
    bomb_result = convolve2d(array == 9, kernel, mode='same', boundary='fill', fillvalue=0)

    # Update the original array by adding 1 for each count from the regular bomb convolution result
    array += bomb_result

    # Clip values to a maximum of 9
    array = np.clip(array, 0, 9)

    return array


def print_board(board):
    print(mine_count)
    for row in board:
        print('\t|'.join(map(str, row)))


class Game(object):
    def __init__(self, rows=9, col=9, per=0.125):
        self._board, self._startx, self._starty = generate_mines(rows, col, per)
        self._board = add_one_to_nines(self._board)

    def show_board(self):
        print_board(self._board)

    def get_board(self):
        return self._board

    def get_start(self):
        return self._startx,self._starty


def main():
    a = Game(9, 9, 0.125)
    a.show_board()


if __name__ == '__main__':
    main()