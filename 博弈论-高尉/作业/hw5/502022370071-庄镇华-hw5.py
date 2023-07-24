from random import choice
from math import inf
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QMessageBox, QLabel, QDialog, QInputDialog
from PyQt5.QtGui import QFont
import sys

class TicTacToe:
    def __init__(self, size=3):
        # 初始化棋盘
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]

    def clear_board(self):
        # 清空棋盘
        for i, row in enumerate(self.board):
            for j, _ in enumerate(row):
                self.board[i][j] = 0

    def check_winner(self, player):
        # 检查行
        for row in self.board:
            if all(cell == player for cell in row):
                return True
        # 检查列
        for col in range(self.size):
            if all(row[col] == player for row in self.board):
                return True
        # 检查对角线
        if all(self.board[i][i] == player for i in range(self.size)):
            return True
        if all(self.board[i][self.size - i - 1] == player for i in range(self.size)):
            return True
        return False

    def is_game_over(self):
        # 检查游戏是否结束
        return self.check_winner(1) or self.check_winner(-1)

    def get_empty_cells(self):
        # 获取空白的格子
        empty_cells = []
        for i, row in enumerate(self.board):
            for j, cell in enumerate(row):
                if cell == 0:
                    empty_cells.append([i, j])
        return empty_cells

    def is_board_full(self):
        # 检查棋盘是否已满
        return len(self.get_empty_cells()) == 0

    def make_move(self, x, y, player):
        # 执行一次移动
        self.board[x][y] = player

    # Heuristic evaluation function E(n) = M(n) - O(n)
    # M(n): the number of my possible winning lines
    # O(n): the number of the oppenent's possible winning lines
    def evaluation(self):
        M_n, O_n = 8, 8
        # 检查行
        for row in self.board:
            if any(cell == -1 for cell in row):
                M_n -= 1
            if any(cell == 1 for cell in row):
                O_n -= 1
            
        # 检查列
        for col in range(self.size):
            if any(row[col] == -1 for row in self.board):
                M_n -= 1
            if any(row[col] == 1 for row in self.board):
                O_n -= 1
        # 检查对角线
        if any(self.board[i][i] == -1 for i in range(self.size)):
            M_n -= 1
        if any(self.board[i][i] == 1 for i in range(self.size)):
            O_n -= 1
        if any(self.board[i][self.size - i - 1] == -1 for i in range(self.size)):
            M_n -= 1
        if any(self.board[i][self.size - i - 1] == 1 for i in range(self.size)):
            O_n -= 1
        return M_n - O_n
    
    
    def get_heuristic_score(self):
        # 获取当前的分数
        if self.check_winner(1):
            return 100
        elif self.check_winner(-1):
            return -100
        else:
            return self.evaluation()

    def alpha_beta_minimax(self, depth, alpha, beta, player):
        # Alpha-Beta pruning的Minimax算法
        if self.is_game_over() or self.is_board_full() or depth == 0:
            return [self.get_heuristic_score()]
        scores = []  # 存储分数
        moves = []  # 存储移动
        # for a in ACTIONS(state) do
        for cell in self.get_empty_cells():
            # v <- MAX(v, MIN-VALUE(Result(s,a), alpha, beta))
            self.make_move(cell[0], cell[1], player)
            score = self.alpha_beta_minimax(depth - 1, alpha, beta, -player)
            scores.append(score[0])
            moves.append(cell)
            self.make_move(cell[0], cell[1], 0)  # 撤销移动
            if player == 1:
                # alpha = MAX(alpha, v)
                if score[0] > alpha:
                    alpha = score[0]
                # if v >= beta then return v
                if alpha >= beta:
                    break
            else:
                if score[0] < beta:
                    beta = score[0]
                if alpha >= beta:
                    break
        if player == 1: # return alpha
            max_score_index = scores.index(max(scores))
            return [scores[max_score_index], moves[max_score_index]]
        else:           # return beta
            min_score_index = scores.index(min(scores))
            return [scores[min_score_index], moves[min_score_index]]

class TicTacToeApp(QWidget):
    def __init__(self, tic_tac_toe):
        super().__init__()
        self.tic_tac_toe = tic_tac_toe
        self.size = self.tic_tac_toe.size
        self.init_ui()
        self.choose_starting_player()

    def init_ui(self):
        # 初始化UI界面
        self.setWindowTitle('基于alpha-beta剪枝的井字游戏')
        self.setFixedSize(self.size*120, self.size*120)
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        self.buttons = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                button = QPushButton()
                button.setFixedSize(100, 100)
                button.setFont(QFont('Arial', 24))
                button.clicked.connect(self.cell_click(i, j))
                self.grid.addWidget(button, i, j)
                row.append(button)
            self.buttons.append(row)

    def choose_starting_player(self):
        dialog = QMessageBox()
        dialog.setWindowTitle('游戏顺序')
        dialog.setText('您要选择先手吗?')
        dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        result = dialog.exec_()
        if result == QMessageBox.No:
            self.player_start = False
            self.ai_move()
        else:
            self.player_start = True
            
    def cell_click(self, i, j):
        # 定义点击事件
        def wrapper():
            if self.buttons[i][j].text() == '':
                self.buttons[i][j].setText('X')
                self.tic_tac_toe.make_move(i, j, 1)
                if self.tic_tac_toe.is_game_over():
                    self.show_win_dialog('您赢了!    ')
                elif not self.tic_tac_toe.is_board_full():
                    self.ai_move()
                else:
                    self.show_win_dialog('达成平局!    ')
        return wrapper

    def ai_move(self):
        # AI的移动
        # depth = 2, player = -1
        _, move = self.tic_tac_toe.alpha_beta_minimax(3, -inf, inf, -1)
        self.tic_tac_toe.make_move(move[0], move[1], -1)
        self.buttons[move[0]][move[1]].setText('O')
        if self.tic_tac_toe.is_game_over():
            self.show_win_dialog('电脑赢了!    ')
        elif self.tic_tac_toe.is_board_full():
            self.show_win_dialog('达成平局!    ')

    def show_win_dialog(self, message):
        # 显示赢家信息
        win_dialog = QMessageBox()
        win_dialog.setWindowTitle('游戏结束')
        win_dialog.setText(message)
        win_dialog.setFixedSize(1000, 400)  # 设置对话框的大小
        win_dialog.exec_()
        self.tic_tac_toe.clear_board()
        for i in range(self.size):
            for j in range(self.size):
                self.buttons[i][j].setText('')
        if not self.player_start:
            self.ai_move()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    tictactoe = TicTacToe()
    ticTacToeApp = TicTacToeApp(tictactoe)
    ticTacToeApp.show()
    sys.exit(app.exec_())
    