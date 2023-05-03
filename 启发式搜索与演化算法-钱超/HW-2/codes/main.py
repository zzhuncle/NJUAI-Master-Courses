import networkx as nx
import numpy as np
import argparse
from matplotlib import pyplot as plt
from EA import GeneticAlgorithm

def generate_regular_graph(args):
    # 这里简单以正则图为例, 鼓励同学们尝试在其他类型的图(具体可查看如下的nx文档)上测试算法性能
    # nx文档 https://networkx.org/documentation/stable/reference/generators.html
    graph = nx.random_graphs.random_regular_graph(d=args.n_d, n=args.n_nodes, seed=args.seed_g)
    return graph, len(graph.nodes), len(graph.edges)

def generate_erdos_renyi_graph(args):
    graph = nx.random_graphs.erdos_renyi_graph(n=args.n_nodes, p=args.prob, seed=args.seed_g)
    return graph, len(graph.nodes), len(graph.edges)

def generate_watts_strogatz_graph(args):
    graph = nx.random_graphs.watts_strogatz_graph(k=args.n_k, n=args.n_nodes, p=args.prob, seed=args.seed_g)
    return graph, len(graph.nodes), len(graph.edges)

def generate_barabasi_albert_graph(args):
    graph = nx.random_graphs.barabasi_albert_graph(n=args.n_nodes, m=args.n_attach, seed=args.seed_g)
    return graph, len(graph.nodes), len(graph.edges)

def generate_gset_graph(args):
    # 这里提供了比较流行的图集合: Gset, 用于进行分割
    dir = './Gset/'
    fname = f'{dir}G{str(args.gset_id)}.txt'
    graph_file = open(fname)
    n_nodes, n_e = graph_file.readline().rstrip().split(' ')
    nodes = list(range(int(n_nodes)))
    edges = []
    for line in graph_file:
        n1, n2, w = line.split(' ')
        edges.append((int(n1) - 1, int(n2) - 1, int(w)))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)
    return graph, len(graph.nodes), len(graph.edges)


def graph_generator(args):
    if args.graph_type == 'regular':
        return generate_regular_graph(args)
    elif args.graph_type == 'ER':
        return generate_erdos_renyi_graph(args)
    elif args.graph_type == 'WS':
        return generate_watts_strogatz_graph(args)
    elif args.graph_type == 'BA':
        return generate_barabasi_albert_graph(args)
    elif args.graph_type == 'gset':
        return generate_gset_graph(args)
    else:
        raise NotImplementedError('Wrong graph_tpye')


def get_fitness(graph, x, n_edges, threshold=0):
    x_eval = np.where(x >= threshold, 1, -1)
    # 获得Cuts值需要将图分为两部分, 这里默认以0为阈值把解分成两块.
    g1 = np.where(x_eval == -1)[0]
    g2 = np.where(x_eval == 1)[0]
    return nx.cut_size(graph, g1, g2) / n_edges

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-type', type=str, help='graph type', default='regular')
    parser.add_argument('--n-nodes', type=int, help='the number of nodes', default=1000) # 可以自己修改
    parser.add_argument('--n-d', type=int, help='the number of degrees for each node', default=10) # 可以自己修改 已有的不要浪费，进行大规模实验验证
    parser.add_argument('--n-k', type=int, help='each node is joined with its k nearest neighbors in a ring topology', default=10) # 可以自己修改 已有的不要浪费，进行大规模实验验证
    parser.add_argument('--n-attach', type=int, help='the number of edges to attach from a new node to existing nodes', default=6) # 可以自己修改 已有的不要浪费，进行大规模实验验证
    parser.add_argument('--prob', type=float, help='the probability for edge creation', default=.012) # 可以自己修改 已有的不要浪费，进行大规模实验验证
    parser.add_argument('--T', type=int, help='the number of fitness evaluations', default=5000)
    parser.add_argument('--seed-g', type=int, help='the seed of generating regular graph', default=1)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--gset-id', type=int, default=1)
    parser.add_argument('--sigma', type=float, help='hyper-parameter of mutation operator',default=.1)
    parser.add_argument('--show-graph', type=bool, help='whether show gragh or not',default=False)
    return parser.parse_known_args()[0]


def main(args=get_args()):
    print(args)
    graph, n_nodes, n_edges = graph_generator(args)
    print(n_nodes, n_edges) # 查看节点数、边数
    ''' 助教的实现
    np.random.seed(args.seed)
    x = np.random.rand(n_nodes)  # 这里x使用实数值表示, 也可以直接使用01串表示, 并使用01串上的交叉变异算子
    best_fitness = get_fitness(graph, x, n_edges)
    for i in range(args.T):  # 简单的(1+1)ES
        tmp = x + np.random.randn(n_nodes) * args.sigma # randn函数返回一个或一组样本，具有标准正态分布。
        tmp_fitness = get_fitness(graph, tmp, n_edges)
        if tmp_fitness > best_fitness:
            x, best_fitness = tmp, tmp_fitness
            print(i, best_fitness) # 0.48696
    '''
    ea = GeneticAlgorithm(args, graph, 100) # 初始化演化算法
    ea.run()
    label = args.graph_type if args.graph_type != 'gset' else f'{args.graph_type}{args.gset_id}'
    print(f'{label} run over.')

if __name__ == '__main__':
    main()
