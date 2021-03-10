import numpy as np

################################################
# 			Layers
################################################
class Input():
	def __init__(self, data):
		self.name = "Input"
		self.input = data.reshape(-1,1)
		self.input = np.append(self.input, 1).reshape(-1,1)
		self.a = self.input
		self.size = self.input.size

class Dense():
	def __init__(self, size, activation, intialization, name):
		self.name = name
		self.size = size
		self.activation = activation
		# Code for initialization
		self.W = 

################################################
# 			Activations
################################################
class Sigmoid():
	def __init__(self, c=1, b=0):
		self.c = c
		self.b = b

	def value(self, x):
		val = 1 + np.exp(-self.c*(x + self.b))
		return 1/val

	def diff(self, x):
		y = self.value(x)
		val = self.c*y*(1-y)
		return val

class Tanh():
	def __init__(self):
		pass

	def value(self, x):
		num = np.exp(x) - np.exp(-x)
		denom = np.exp(x) + np.exp(-x)
		return num/denom

	def diff(self, x):
		y = self.value(x)
		val = 1 - y**2
		return val

class Relu():
	def __init__(self):
		pass

	def value(self, x):
		val = x
		val[val<0] = 0
		return val

	def diff(self, x):
		val = np.ones(x.shape)
		val[val<=0] = 0
		return val

class Softmax():
	def __init__(self):
		pass

	def value(self, x):
		val = np.exp(x)/np.sum(np.exp(x))
		return val

	def diff(self, x):
		y = self.value(x)
		# Motivation for condensed equation:
		# https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
		val = (np.ones(y.shape) - y)
		return val

################################################
# 			Optimizers
################################################
class Momentum():
	def __init__(self, gamma=None):
		self.v = 0
		self.gamma = gamma

	def get_update(self, W, grad, eta):
		v = self.gamma*self.v + eta*grad
		W = W - v
		self.v = v
		return W

# Doesn't work
# class NAG():
# 	def __init__(self,gamma=None):
# 		self.update = 0
# 		self.gamma = gamma

# 	def get_update(self, W, grad, eta):
# 		W_lookahead = W - self.gamma*self.update
# 		update = self.gamma + eta*grad

class AdaGrad():
	def __init__(self):
		self.v = 0
	
	def get_update(self, W, eta, grad, eps=1e-7):
		# eps value as in keras
		v = self.v + np.linalg.norm(grad)**2
		W = W - (eta/(v+eps)**0.5)*grad
		self.v = v
		return W

class RMSProp():
	def __init__(self, beta=None):
		self.v = 0
		self.beta = beta

	def get_update(self, W, eta, grad, eps=1e-7):
		v = self.beta*self.v + (1-self.beta)*np.linalg.norm(grad)**2
		W = W - (eta*(v+eps)**0.5)*grad

class Adam():
	def __init__(self, beta1=None, beta2=None):
		self.m = 0
		self.v = 0
		self.m_cap = 0
		self.v_cap = 0
		self.beta1 = beta1
		self.beta2 = beta2

	def get_iteration(self):
		if self.m == 0 and self.m_cap == 0:
			return 1
		else:
			iteration = int(np.ceil(np.log(1-(self.m/self.m_cap))/np.log(self.beta1)))
			return iteration

	def get_update(self, W, eta, grad, eps=1e-7):
		iteration = self.get_iteration()
		m = self.beta1*self.m + (1-self.beta1)*grad
		v = self.beta2*self.v + (1-self.beta2)*np.linalg.norm(grad)**2
		m_cap = m/(1-self.beta1**t)
		v_cap = v/(1-self.beta2**t)

		W = W - (eta/(v_cap+eps)**0.5)*m_cap
		return W

################################################
# 			Network
################################################

class NeuralNetwork():
	def __init__(self, layers, loss, batch_size):
		self.layers = layers
		self.batch_size = batch_size
		self.loss = loss

	def forward_propogation(self, epochs):
		X = self.layers[0].input
		for i in range(1, len(self.layers)):
			self.layers[i].h = self.layers[i].W @ self.layers[i-1].a + self.layers[i].b
			self.layers[i].a = self.layers[i].activation.value(self.layers[i].h)

	def backward_propogation(self)