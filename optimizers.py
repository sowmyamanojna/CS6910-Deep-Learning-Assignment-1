import numpy as np

################################################
#         Optimizers
################################################
class Momentum():
    def __init__(self, eta=None, gamma=None):
        self.update = 0
        self.eta = eta
        self.gamma = gamma

    def get_update(self, W, grad):
        self.update = self.gamma*self.update + self.eta*grad
        W = W - self.update
        return W

class Nesterov():
    def __init__(self, eta=None, gamma=None):
        self.update = 0
        self.eta = eta
        self.gamma = gamma
        
    def get_update(self, W):
        # Have to still work on this
        W_lookahead = W - self.gamma*self.update
        self.update = self.gamma*self.update + self.eta*gradient(W_lookahead) # Need to call gradient function
        W = W - self.update
        return W
        

class AdaGrad():
    def __init__(self, eta=1e-3, eps=1e-7):
        self.v = 0
        self.eta = eta
        self.eps = eps
    
    def get_update(self, W, grad):
        self.v = self.v + grad**2
        W = W - (self.eta/(self.v+self.eps)**0.5)*grad
        return W

class RMSProp():
    def __init__(self, beta=0.9, eta = 1e-3, eps = 1e-7):
        self.v = 0
        self.beta = beta
        self.eta = eta
        self.eps = eps

    def get_update(self, W, grad):
        self.v = self.beta*self.v + (1-self.beta)*(grad**2)
        W = W - (self.eta/(self.v+self.eps)**0.5)*grad
        return W

class Adam():
    def __init__(self, beta1=0.9, beta2=0.999, eta=1e-3, eps=1e-7):
        self.m = 0
        self.v = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.eps = eps
        self.iter = 1

    def get_update(self, W, grad):
        self.m = self.beta1*self.m + (1-self.beta1)*grad
        self.v = self.beta2*self.v + (1-self.beta2)*(grad**2)
        m_cap = self.m/(1-self.beta1**self.iter)
        v_cap = self.v/(1-self.beta2**self.iter)        
        W = W - (self.eta/(v_cap+self.eps)**0.5)*m_cap
        self.iter += 1
        return W

class Nadam():
    # Reference: https://ruder.io/optimizing-gradient-descent/index.html#nadam
    def __init__(self, beta1=0.9, beta2=0.999, eta=1e-3, eps=1e-7):
        self.m = 0
        self.v = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.eps = eps
        self.iter = 1
    
    def get_update(self, W, grad):
        self.m = self.beta1*self.m + (1-self.beta1)*grad
        self.v = self.beta2*self.v + (1-self.beta2)*(grad**2)
        m_cap = self.m/(1-self.beta1**self.iter)
        v_cap = self.v/(1-self.beta2**self.iter) 
        update = self.beta1*m_cap + ((1-self.beta1)/(1-self.beta1**self.iter))*grad
        W = W - (self.eta/(v_cap+self.eps)**0.5)*update
        self.iter += 1
        return W