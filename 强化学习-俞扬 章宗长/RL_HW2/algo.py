import numpy as np
from abc import abstractmethod
from collections import defaultdict
import random

class QAgent:
	def __init__(self,):
		pass

	@abstractmethod
	def select_action(self, ob):
		pass

class QLearningAgent(QAgent):
	def __init__(self):
		super().__init__()
		self.gamma = 0.99
		self.qtable = defaultdict(lambda : [0., 0., 0., 0.])

	def select_action(self, ob):
		o = str(ob[0]) + str(ob[1])
		all_q = np.array(self.qtable[o])
		idxes = np.argwhere(all_q == np.max(all_q))
		return random.choice(idxes)[0]

	def update(self, ob, a, r, next_ob, step):
		o = str(ob[0]) + str(ob[1])
		next_o = str(next_ob[0]) + str(next_ob[1])
		next_a = self.select_action(next_o)
		old_q = self.qtable[o][a]
		new_q = r + self.gamma * self.qtable[next_o][next_a]
		self.qtable[o][a] += self.get_lr(step) * (new_q - old_q)

	def get_lr(self, step):
		if step <= 2000:
			return 0.1
		elif step <= 10000:
			return 1
		elif step <= 20000:
			return 0.01
		else:
			return 0.001