# 复现报告中的结果
1. 切换到 code 文件夹，文件夹中包含 EA.py （演化算法实现）和 main.py （调用演化算法）。
    ```shell
    cd code
    ```
2. 关于基础和改进版算法的演化算子选择，可以更改 EA.py 文件中的以下部分实现。
    ```python
    # 基础版
    class GeneticAlgorithm(object):
        def __init__(self, args, graph, population_size):
            self.variation_operator = uniform_crossover
            self.mutation_operator = bitwise_mutation
            self.selection_operator = tournament_selection
            self.parents_operator = self.uniform_parents
    ```
    ```python
    # 改进版
    class GeneticAlgorithm(object):
        def __init__(self, args, graph, population_size):
            self.variation_operator = uniform_crossover
            self.mutation_operator = onebit_mutation
            self.selection_operator = best_selection
            self.parents_operator = self.fitness_parents
    ```
3.  run.bash 可以同时运行演化算法在 8 张图上的实验，其命令如下，运行后会生成 适应度的 numpy 文件，可供画图调用。
    ```bash
    #!/bin/bash
    python main.py --graph-type=gset --gset-id=1 > out1.txt 2>&1 &
    python main.py --graph-type=gset --gset-id=2 > out2.txt 2>&1 &
    python main.py --graph-type=gset --gset-id=3 > out3.txt 2>&1 &
    python main.py --graph-type=gset --gset-id=4 > out4.txt 2>&1 &
    python main.py --graph-type=regular > out5.txt 2>&1 &
    python main.py --graph-type=ER > out6.txt 2>&1 &
    python main.py --graph-type=WS > out7.txt 2>&1 &
    python main.py --graph-type=BA > out8.txt 2>&1 &
    ```
