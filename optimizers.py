import numpy as np

################################################
#         Optimizers
################################################
class Momentum():
    def __init__(self, eta=None, gamma=None):
        self.update = 0
        self.eta = eta
        self.gamma = gamma

    def get_update(self, grad):
        self.update = self.gamma*self.update + self.eta*grad
        return self.update

class Nesterov():
    def __init__(self, eta=None, gamma=None):
        self.update = 0
        self.eta = eta
        self.gamma = gamma
        
    def get_update(self, grad=None):
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
    
    def get_update(self, grad):
        self.v = self.v + grad**2
        return (self.eta/(self.v+self.eps)**0.5)*grad

class RMSProp():
    def __init__(self, beta=0.9, eta = 1e-3, eps = 1e-7):
        self.v = 0
        self.beta = beta
        self.eta = eta
        self.eps = eps

    def get_update(self, grad):
        self.v = self.beta*self.v + (1-self.beta)*(grad**2)
        return (self.eta/(self.v+self.eps)**0.5)*grad

class Adam():
    def __init__(self, beta1=0.9, beta2=0.999, eta=1e-3, eps=1e-8):
        self.m = 0
        self.v = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.eps = eps
        self.iter = 1

    def get_update(self, grad):
        # print("Grad shape:", ((1-self.beta1)*grad).shape)
        # try:
        #     print("(from try) m Shape:", (self.beta1*self.m).shape)
        # except:
        #     pass
        self.m = self.beta1*self.m + (1-self.beta1)*grad
        self.v = self.beta2*self.v + (1-self.beta2)*(grad**2)
        # print("Shape change?:", self.m.shape)
        m_cap = self.m/(1-self.beta1**self.iter)
        v_cap = self.v/(1-self.beta2**self.iter)        
        self.iter += 1
        return (self.eta/(v_cap+self.eps)**0.5)*m_cap

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
    
    def get_update(self, grad):
        self.m = self.beta1*self.m + (1-self.beta1)*grad
        self.v = self.beta2*self.v + (1-self.beta2)*(grad**2)
        m_cap = self.m/(1-self.beta1**self.iter)
        v_cap = self.v/(1-self.beta2**self.iter) 
        update = self.beta1*m_cap + ((1-self.beta1)/(1-self.beta1**self.iter))*grad
        self.iter += 1
        return (self.eta/(v_cap+self.eps)**0.5)*update