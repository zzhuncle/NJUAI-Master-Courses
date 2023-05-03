import numpy as np
from abc import abstractmethod
from sklearn.neighbors import KNeighborsClassifier
import pickle
# 使用最近邻模型
clf = KNeighborsClassifier(n_neighbors = 10, metric = "minkowski")
import joblib # 保存sklearn模型

class DaggerAgent:
	def __init__(self,):
		pass

	@abstractmethod
	def select_action(self, ob):
		pass

# here is an example of creating your own Agent
class ExampleAgent(DaggerAgent):
	def __init__(self, path = './model.pkl'):
		super(DaggerAgent, self).__init__()
		# init your model
		self.model = clf
		self.model = joblib.load(path)

	# train your model with labeled data
	def update(self, data, label):
     	# self.model.train(data_batch, label_batch)
		clf.fit(data, label)
		self.save() # save model

	# select actions by your model
	def select_action(self, data_batch):
		pred_batch = clf.predict(data_batch)
		return pred_batch

	# save your model in specific path
	def save(self, path = './model.pkl'):
		joblib.dump(self.model, path)
		# my_model = joblib.load(path) 读取模型
