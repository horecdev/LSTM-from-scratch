
import numpy as np

class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0 # Start the counter of iterations
        
        # Initialize the m and v for every param
        self.m = [np.zeros_like(p) for p, g in self.params]
        self.v = [np.zeros_like(p) for p, g in self.params]

    def step(self, lr=None):
        self.t += 1
        current_lr = lr if lr is not None else self.lr

        for i, (p, grad) in enumerate(self.params):
            # Because these are EMAs we exponentialy forget the past
            # We give more weight to recent events

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad 
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # The m tracks "what direction were recent grads?" And if they were jittering back
            # and forth then they will be around 0
            # The v tracks the magnitude of the updates - the square removes sign. Also gives more memory to the present than past
            

            # Bias correction (accounts for the fact that m, v start at 0)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t) # divided by small number when t small, by 1 when t big
            v_hat = self.v[i] / (1 - self.beta2 ** self.t) # same story

            p -= self.current_lr * m_hat / (np.sqrt(v_hat) + self.eps) # accounts for direction (m_hat) and intensity (v_hat)
            # The n_hat works as momentum of direction, and v_hat as scaling if gradients are huge or tiny.
            
    def get_state(self):
        return {
            "m" : self.m,
            "v" : self.v,
            "t" : self.t
        }
        
    def load_state(self, state_dict):
        self.m = state_dict["m"]
        self.v = state_dict["v"]
        self.t = state_dict["t"]