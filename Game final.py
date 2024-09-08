import numpy as np
import numba as nb
import copy
import math
from IPython.display import clear_output
import time
from numpy.lib.stride_tricks import sliding_window_view

import tkinter as tk
from tkinter import *

gameType = 'pvc'
boardSize = 9
winCount = 5

# playerPosX
# playerPosY

class Board:
    global gameType, boardSize, winCount

    def __init__(self, keyList):
        self.board = np.zeros((boardSize, boardSize), dtype = int)
        self.keyList = keyList
        self.winner = None

    def get_board(self):
        return self.board

    def check_empty(self, x, y, pos):
        return self.board[x][y] == pos

    def place_key(self, x, y, key):
        self.board[x][y] = key

    @staticmethod
    def get_empty(state):
        empty = []
        for i in range(boardSize):
            for j in range(boardSize):
                if state[i][j] == 0:
                    empty.append((i, j,))
        return empty

    def check_status(self, state):

        checkList = []
        noEmpty = len(self.get_empty(state)) == 0

        index = 0

        for i in range(-(boardSize - winCount), boardSize - (boardSize - winCount)):
            checkList.append(state[i, :].tolist())
            checkList.append(state[:, i].tolist())
            index += 1

            checkList.append(state.diagonal(i).tolist())
            checkList.append(state[::-1].diagonal(i).tolist())

        checkList = [''.join(np.char.mod('%d', checkList[i])) for i in range(len(checkList))]

        for i in range(len(checkList)):
            if str(self.keyList[0]) * winCount in checkList[i]:
                print('human player', 'win', 'in', checkList[i])
                return self.keyList[0]

            elif str(self.keyList[1]) * winCount in checkList[i]:
                print('ai player', 'win', 'in', checkList[i])
                return self.keyList[1]

        if noEmpty:
            return -1

        return 0


class Evaluation:
    global gameType, boardSize, winCount

    def __init__(self):
        self.rates = {1: self.rate('1'), 2: self.rate('2')}

    @staticmethod
    def rate(x):
        return [
            [f'{x * 5}', 5000000000],
            [f'0{x * 3}0', 100000000],
            [f'{x * 4}0', f'0{x * 4}', f'{x * 2}0{x * 2}', f'{x}0{x * 3}', f'{x * 3}0{x}', 1000000],
            [f'0{x * 3}0', f'{x}0{x * 2}0', f'0{x * 2}0{x}', f'{x}0{x * 2}0', f'00{x * 3}', f'{x * 3}00', 100000],
            [f'000{x * 2}', f'{x * 2}000', f'{x}0{x}00', f'00{x}0{x}', 10000],
            [f'{x}', 1]
        ]

    # @nb.jit()
    def evaluate(self, board, player):
        # print(board, end = '')
        board = np.char.mod('%d', board)
        checkList = board.tolist()
        checkList.extend(board.T.tolist())
        for i in range(-boardSize + 2, boardSize - 1):

            if len(board.diagonal(i).tolist()) >= winCount:
                checkList.append(board.diagonal(i).tolist())
                checkList.append(np.fliplr(board).diagonal(i).tolist())

        point = 0
        for i in range(len(checkList)):
            sliding_window = sliding_window_view(checkList[i], window_shape = winCount)
            for s in sliding_window:
                for v in self.rates[player]:
                    if ''.join(s) in v:
                        point += v[-1]
        # print(point, player)
        if player == 2:
            return -point

        return point


class Player:
    def __init__(self, key, board):
        self.key = key
        self.board = board


class HumanPlayer(Player):
    global gameType, boardSize, winCount, tv, posX, posY

    def move(self):

        x, y = int(posX / 40 + 0.5) - 1, int(posY / 40 + 0.5) - 1
        print(x, y)
        # switch from image coordinate to chessboard coordinate
        if x < boardSize and y < boardSize and self.board.check_empty(x, y, 0):
            self.board.place_key(x, y, self.key)
            # print(self.board.get_board())
            print_chess(self)
            # print the chess on the
            return True
        else:
            print("invalid move")
            return False


class AiPlayer(Player):
    global gameType, boardSize, winCount

    def __init__(self, key, board):
        Player.__init__(self, key, board)
        self.searchDepth = 2
        self.board = board
        self.evaluator = Evaluation()

        weight = boardSize // 2
        self.weightList = np.array([weight - 1, weight - 1, weight - 1,
                                    weight - 1, weight, weight - 1,
                                    weight - 1, weight - 1, weight - 1, ]).reshape(3, 3)

        for i in range(1, weight):
            horizontalWeights = np.array([weight - i] * (2 * i + 1)).reshape((2 * i + 1), 1)
            verticalWeights = np.array([weight - i] * (2 * i + 3)).reshape(1, (2 * i + 3))

            self.weightList = np.column_stack((self.weightList, horizontalWeights))
            self.weightList = np.column_stack((horizontalWeights, self.weightList))

            self.weightList = np.row_stack((verticalWeights, self.weightList))
            self.weightList = np.row_stack((self.weightList, verticalWeights))

    # @nb.jit()
    def minimax(self, state, depth, alpha, beta, player):


        if depth == 0 or self.board.check_status(state) != 0:
            point = self.evaluator.evaluate(state, (player % 2 + 1))
            print(state)
            return point, None

        if player == 1:
            maxEval = -math.inf
            # empty = np.sort(np.array(self.board.get_empty(state), dtype = [('x', int), ('y', int)]), order = ['x', 'y'])[::-1]
            empty = self.board.get_empty(state)
            # np.random.shuffle(empty)
            posX1, posY1 = empty[len(empty) // 2][0], empty[len(empty) // 2][1]
            for x, y in empty:
                state[x][y] = 1
                evaluation, _ = self.minimax(state, depth - 1, alpha, beta, 2)
                evaluation += self.weightList[x][y]
                state[x][y] = 0

                if evaluation > maxEval:
                    # maxEval = evaluation
                    posX1 = x
                    posY1 = y

                maxEval = max(maxEval, evaluation)
                alpha = max(alpha, evaluation)

                if beta <= alpha:
                    break
            return maxEval, [posX1, posY1]
        else:
            minEval = math.inf
            # empty = np.sort(np.array(self.board.get_empty(state), dtype = [('x', int), ('y', int)]), order = ['x', 'y'])[::-1]
            empty = self.board.get_empty(state)
            #np.random.shuffle(empty)
            posX1, posY1 = empty[len(empty) // 2][0], empty[len(empty) // 2][1]
            for x, y in empty:
                state[x][y] = self.key  # if self.key == 1 else 1
                evaluation, _ = self.minimax(state, depth - 1, alpha, beta, 1)
                evaluation += self.weightList[x][y]
                state[x][y] = 0

                if evaluation < minEval:
                    # minEval = evaluation
                    posX1 = x
                    posY1 = y

                minEval = min(minEval, evaluation)
                beta = min(beta, evaluation)

                if beta <= alpha:
                    break
            return minEval, [posX1, posY1]



    def move(self):
        _, bestPlace = self.minimax(copy.deepcopy(self.board.get_board()), self.searchDepth, -math.inf, math.inf, 2)
        self.board.place_key(bestPlace[0], bestPlace[1], self.key)
        print_chess(self)

        return True


# board = np.zeros([15, 15], dtype=int)
i, j = 0, 0

window = tk.Tk()
tv = tk.Canvas(window, height = 400, width = 400, bg = "#cfa340")

#######################################################################

keyList = [1, 2]

currentPlayer = 0  # 1 = 电脑先
board = Board(keyList)
player1 = HumanPlayer(keyList[abs(currentPlayer)], board)
ai = AiPlayer(keyList[abs(currentPlayer - 1)], board)
players = [player1, ai]
active = 0

window.title('Game')
window.geometry(str(1000 - 200) + 'x' +
                str(1000 - 200) + '+' +
                str(3024 // 10) + '+' +
                str((1964 - 1000) // 10))
window.resizable(False, False)
for i in range(8):
    coord = 40, 40, 360, i * 40 + 80
    tv.create_rectangle(coord)
    coord = 40, 40, i * 40 + 80, 360
    tv.create_rectangle(coord)

posX, posY = 5000, 5000

def print_chess(self):
    color = ['white', 'black']
    for i in range(len(self.board.get_board())):
        for j in range(len(self.board.get_board())):
            if self.board.get_board()[i][j] != 0:
                tv.create_oval(((i + 1) * 40) + 12, ((j + 1) * 40) + 12, ((i + 1) * 40) - 12,
                               ((j + 1) * 40) - 12,
                               fill = color[self.board.get_board()[i][j] - 1],
                               outline = color[self.board.get_board()[i][j] - 1])

def left_click(event):
    global posX, posY
    if window.title() == "Game":
        posX = event.x
        posY = event.y

def quit_game():
    window.destroy()


window.bind("<Button-1>", left_click)
tv.pack(side = LEFT, padx = 30)
quitButton = Button(window, text = ' Quit ', height = 3, width = 20,
                    command = quit_game)
quitButton.pack(side = BOTTOM, pady = 50)

while active == 0:
    if players[currentPlayer].move():
        currentPlayer = abs(currentPlayer - 1)

    active = board.check_status(board.get_board())
    tv.update_idletasks()
    tv.update()
