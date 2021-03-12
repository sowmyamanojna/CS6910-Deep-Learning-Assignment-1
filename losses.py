import numpy as np

################################################
#         Losses
################################################
class CrossEntropy():
	def __init__(self):
		pass

	def calc_loss(self, t, y):
		loss = np.sum(np.sum(t*np.log(y)))

	
