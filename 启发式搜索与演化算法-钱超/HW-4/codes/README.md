# 复现报告中的结果
1. 切换到 code 文件夹，文件夹中包含8个ipynb文件，分别实现四种方法在稀疏回归和最大覆盖问题上的实验。
    ```shell
    run POSS_Regression.ipynb # 实现POSS算法在稀疏回归问题上的实验
    run POSS_MaxCut.ipynb # 实现POSS算法在最大覆盖问题上的实验
    运行其余三种方法命令类似于POSS算法
    e.g. run DPOSS_MaxCut.ipynb # 实现改进的DPOSS算法在最大覆盖问题上的实验
    ```
    
2. 关于稀疏回归和最大覆盖问题的数据集选择，可以更改ipynb文件中的以下部分实现。
    ```python
    # 稀疏回归问题
    from sklearn.datasets import load_svmlight_file
    
    data_path = 'svmguide3' # 这里可以选择sonar数据集和svmguide3数据集
    data = load_svmlight_file(f'../datasets/{data_path}.txt')
    x = data[0].todense()
    y = data[1].reshape(-1,1)
    ```
    ```python
    # 最大覆盖问题
    import networkx as nx
    import numpy as np
    
    def generate_regular_graph():
        graph = nx.random_graphs.random_regular_graph(d=99, n=200, seed=2023)
        return graph, len(graph.nodes), len(graph.edges)
    
    def generate_erdos_renyi_graph():
        graph = nx.random_graphs.erdos_renyi_graph(n=180, p=0.52, seed=2023)
        return graph, len(graph.nodes), len(graph.edges)
    
    data_path = 'regular' # 这里可以选择正则图和ER图
    
    if data_path == 'regular':
        graph, n_nodes, n_edges = generate_regular_graph()
    elif data_path == 'ER':
        graph, n_nodes, n_edges = generate_erdos_renyi_graph()
    ```
