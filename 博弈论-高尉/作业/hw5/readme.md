# Tic Tac Toe Game

这个程序实现了基于 Alpha-Beta 剪枝算法的井字游戏。玩家可以选择先手或后手，与电脑对战。游戏使用 PyQT5 框架进行图形界面展示。

## 游戏规则

- 游戏界面是一个 3 × 3 的棋盘，玩家和电脑轮流落子，玩家执 X，电脑执 O。
- 玩家和电脑交替进行移动，每次移动选择一个空白的格子进行落子。
- 当有任一玩家在行、列或对角线上连成 3 个相同的棋子时，游戏结束，该玩家获胜。
- 如果棋盘填满且没有任何玩家获胜，游戏以平局结束。

## 游戏界面

游戏界面是一个 3 × 3 的棋盘，使用按钮表示每个格子，玩家通过点击按钮进行移动。每次移动后，电脑将根据 Alpha-Beta 剪枝算法进行移动。

## Alpha-Beta 剪枝算法

Alpha-Beta 剪枝算法是一种在博弈树搜索中用于减少节点搜索数量的优化技术。它基于最优值和最坏值的概念，通过剪枝（即丢弃某些无需再搜索的节点）来减少搜索空间，从而提高搜索效率。

### 算法原理

Alpha-Beta 剪枝算法是基于 Minimax 算法的改进，它通过维护两个值来实现剪枝：`alpha` 和 `beta`。其中 `alpha` 表示当前节点的最大值，`beta` 表示当前节点的最小值。

在搜索过程中，如果某个节点的值已经超过了 `alpha` 或 `beta` 的取值范围，则可以放弃对该节点的搜索，因为该节点的上级节点不会选择它。这样可以减少搜索的分支，提高搜索效率。

### 算法步骤

1. 在最大值节点上，初始化 `alpha` 为负无穷大，`beta` 为正无穷大。
2. 在最小值节点上，初始化 `alpha` 为负无穷大，`beta` 为正无穷大。
3. 对于当前节点的每个子节点：
   - 如果当前节点是最大值节点，则递归调用 Alpha-Beta 剪枝算法，将当前节点的 `alpha` 作为子节点的 `alpha`，并更新当前节点的 `alpha`。
   - 如果当前节点是最小值节点，则递归调用 Alpha-Beta 剪枝算法，将当前节点的 `beta` 作为子节点的 `beta`，并更新当前节点的 `beta`。
   - 如果当前节点的 `alpha` 大于等于 `beta`，则剪枝，停止搜索。
4. 根据搜索结果选择最优的子节点。

### 算法特点

Alpha-Beta 剪枝算法具有以下特点：

- 在搜索过程中，通过剪枝减少了搜索空间，提高了搜索效率。
- 只需对子节点进行评估，而不需要评估所有可能的节点。
- 在最好情况下，时间复杂度为 `O(b^(d/2))`，其中 `b` 是分支因子，`d` 是树的深度。
- Alpha-Beta 剪枝算法可以应用于各种博弈类问题，如井字游戏、国际象棋等。

通过使用 Alpha-Beta 剪枝算法，可以在井字游戏等复杂问题中更有效地搜索最优解，提高游戏的智能程度和用户体验。
