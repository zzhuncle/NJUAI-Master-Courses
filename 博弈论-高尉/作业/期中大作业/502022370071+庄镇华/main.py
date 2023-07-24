import numpy as np
from fractions import Fraction
# output as a fraction
np.set_printoptions(formatter={'all':lambda x: str(Fraction(x).limit_denominator())})
import re
import os
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection

# read the strategic game (.nfg) file in Gambit format
def read_info(in_path):
    with open(in_path) as f:
        lines = f.readlines()
        payoffs, players = lines[-1].strip().split(), lines[-3]
    all_payoffs, action_shapes = [], [int(_) for _ in re.search('{ (\d+ )*}', players).group()[1:-1].split()]
    for player in range(len(action_shapes)):
        payoff = np.array([int(_) for i, _ in enumerate(payoffs) if i % len(action_shapes) == player]).reshape(action_shapes, order='F')
        all_payoffs.append(payoff)
    return np.array(all_payoffs), action_shapes

# compute an interior point using linear programming
def compute_interior_point(halfspaces):
    # https://docs.scipy.org/doc//scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),(halfspaces.shape[0], 1))
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = - halfspaces[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b)
    # Will yield a point x that is furthest inside the convex polyhedron. 
    # To be precise, it is the center of the largest hypersphere of radius y inscribed in the polyhedron.
    x, y = res.x[:-1], res.x[-1] 
    return x

# compute halfspace based on payoff matrix
def calc_halfspace(payoff, type):
    # Mx + b <= 0, [M; b]
    m, n = payoff.shape
    if type == 'P':
        b = np.append(np.zeros(n), -np.ones(m))
        M = np.append(-np.eye(n), payoff, axis=0)
    if type == 'Q':
        b = np.append(-np.ones(m), np.zeros(n))
        M = np.append(payoff, -np.eye(n), axis=0)
    return np.column_stack((M, b.transpose()))

# compute vertices and labels
def calc_vertices_and_labels(halfspaces):
    interior_point = compute_interior_point(halfspaces)
    hs = HalfspaceIntersection(halfspaces, interior_point)
    res = []
    for vertex in hs.intersections:
        if not np.all(np.isclose(vertex, 0)):
            A, b = halfspaces[:, :-1], halfspaces[:, -1]
            label = np.where(np.isclose(np.dot(A, vertex), -b))[0]
            res.append((vertex, label))
    return res

# two players, solve all pure and mixed strategy equilibrium via labeled polytopes
def calc_mne(all_payoffs):
    res = []
    A, B = all_payoffs
    # make sure A and B are nonnegative to eliminate the u and v
    if np.min(A) < 0:
        A = A + abs(np.min(A))
    if np.min(B) < 0:
        B = B + abs(np.min(B))
    
    m, n = A.shape
    P_halfspace = calc_halfspace(B.transpose(), 'P')
    Q_halfspace = calc_halfspace(A, 'Q')
    P_vertices = calc_vertices_and_labels(P_halfspace)
    Q_vertices = calc_vertices_and_labels(Q_halfspace)
    
    for x_vertex, x_label in P_vertices:
        labels = np.zeros(m + n, dtype=int)
        labels[x_label] = 1
        for y_vertex, y_label in Q_vertices:
            labels[y_label] += 1
            if np.all(labels):
                res.append(np.append(x_vertex / sum(x_vertex), y_vertex / sum(y_vertex)))
            labels[y_label] -= 1
    return res

# more than two players, solve all pure strategy equilibrium
def calc_pne(all_payoffs):
    player_num, action_shapes, pnes = len(all_payoffs), all_payoffs[0].shape, set()
    for i, payoff in enumerate(all_payoffs):
        cur_pnes = set()
        max_pos = list(np.where(payoff == np.max(payoff, axis=i, keepdims=True)))
        cur_pnes_num = len(max_pos[0])
        for j in range(cur_pnes_num):
            temp = []
            for k in range(player_num):
                temp.append(max_pos[k][j])
            cur_pnes.add(tuple(temp))
        pnes = cur_pnes if i == 0 else pnes.intersection(cur_pnes)

    nes = []
    for pne in pnes:
        ne = []
        for _ in range(player_num):
            cur = [0] * action_shapes[_]
            cur[pne[_]] = 1
            ne += cur
        nes.append(ne)
    return nes

# write nash equilibrium to a file
def write_res(nes, out_path):
    with open(out_path, 'w') as f:
        for ne in nes:
            line = ''
            for i, v in enumerate(ne):
                v = str(Fraction(v).limit_denominator())
                line += v if i == 0 else ',' + v
            f.write(line + '\n')

# calculate nash equilibrium
def nash(in_path, out_path):
    # load file
    all_payoffs, action_shapes = read_info(in_path)
    # get NE
    nes = calc_mne(all_payoffs) if len(action_shapes) == 2 else calc_pne(all_payoffs)
    # write file
    write_res(nes, out_path)


if __name__ == '__main__':
    for f in os.listdir('input'):
        if f.endswith('.nfg'):
            nash('input/'+f, 'output/'+f.replace('nfg','ne'))
            
'''
if __name__ == '__main__':
    for f in os.listdir('examples'):
        if f.endswith('.nfg'):
            nash('examples/'+f, 'examples_output/'+f.replace('nfg','ne'))
'''
