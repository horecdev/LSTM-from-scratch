import numpy as np
import numpy.typing as npt

Tensor = npt.NDArray[np.float64]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Embedding:
    pass

class SoftmaxCrossEntropy:
    pass

class MSELoss:
    pass 

class LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.W_f:   Tensor = np.random.randn(input_dim + hidden_dim, hidden_dim)
        self.W_i:   Tensor = np.random.randn(input_dim + hidden_dim, hidden_dim)
        self.W_can: Tensor = np.random.randn(input_dim + hidden_dim, hidden_dim)
        self.W_o:   Tensor = np.random.randn(input_dim + hidden_dim, hidden_dim)
        
        self.b_f:   Tensor = np.zeros((hidden_dim))
        self.b_i:   Tensor = np.zeros((hidden_dim))
        self.b_can: Tensor = np.zeros((hidden_dim))
        self.b_o :  Tensor = np.zeros((hidden_dim))
        
        self.input_cache: Tensor | None = None
    
    # f_t = sigmoid(W_f * [h_t-1, x_t] + bf) # What to keep from previous cell state
    # i_t = sigmoid(W_i * [h_t-1, x_t] + bo) # What to keep from candidate cell state
    # can_t = tanh(W_can * [h_t-1, x_t] + bcan) # What is candidate cell state
    # o_t = sigmoid(W_o * [h_t-1, x_t] + bo) # What to put into hidden state from cell state
    
    # c_t = f_t * c_t-1 + i_t * can_t # What is remaining from previous cell state + what comes from candidate cell state
    # h_t = o_t * tanh(c_t) # What to put as hidden state from cell_state
    # h_t is like a filtered output - just a part of conveyor belt c_t
    
    
    
    def forward(self, x: Tensor, init_states: tuple[Tensor, Tensor]) -> Tensor: # init_states[0] is h_prev, [1] is c_prev
        # We init both h_prev and c_prev to not lobotomize the model every batch when f.i doing NLP
        # We assume x is (B, seq_len, input_dim)
        self.input_cache = x
        
        B, seq_len, _ = x.shape
        
        if init_states is None:
            h_prev = np.zeros((B, self.hidden_dim))
            c_prev = np.zeros((B, self.hidden_dim))
            
        for t in range(seq_len):
            x_t = x[:, t, :] # (B, input_dim)
            combined_input = np.concatenate([h_prev, x_t], axis=1) # (B, hidden_dim + input_dim)
            f_t = sigmoid()
            
    
    def backward(self):
        pass
    
    def step(self):
        pass