import numpy as np

################################################
#         Loss
################################################
class CrossEntropy():
	def __init__(self):
		pass

	def calc_loss(self, t, y):
		self.t = t
		self.y = y
		loss = -np.sum(np.sum(self.t*np.log(self.y)))
		return loss

	def diff(self):
		grad = -self.t/(self.y)
		return grad

class SquaredError():
	def __init__(self):
		pass

	def calc_loss(self, t, y):
		self.t = t
		self.y = y
		loss = np.sum((t-y)**2)
		return loss

	def diff(self):
		grad = -(self.t - self.y)
		return grad
