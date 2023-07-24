from gurobipy import *
import numpy as np

# 修改系数
maxc = [2, 1, 3]
geqr1 = [1, 0, 4]
geqr2 = [0, 1, 3]
geql = [2, 1, 0]

model = Model('LP1')
# 变量
p1 = model.addVar(lb = 0, ub = 1, vtype = GRB.CONTINUOUS, name = 'p1')
p2 = model.addVar(lb = 0, ub = 1, vtype = GRB.CONTINUOUS, name = 'p2')
model.update()
# 目标函数
model.setObjective(maxc[0]*p1+maxc[1]*p2+maxc[2]*(1-p1-p2), GRB.MAXIMIZE)
# 约束
model.addConstr(geql[0]*p1+geql[1]*p2+geql[2]*(1-p1-p2)>=geqr1[0]*p1+geqr1[1]*p2+geqr1[2]*(1-p1-p2), name = 'c1')
model.addConstr(geql[0]*p1+geql[1]*p2+geql[2]*(1-p1-p2)>=geqr2[0]*p1+geqr2[1]*p2+geqr2[2]*(1-p1-p2), name = 'c2')
model.addConstr(p1+p2>=0, name = 'c3')
model.addConstr(p1+p2<=1, name = 'c4')
# 求解
model.setParam('outPutFlag', 0)
model.optimize()
# 输出
if model.Status == 3:
    print('求解失败')
else:
    print(model.objVal)
    p = [(model.getVars()[i]).x for i in range(2)]
    print(p)
