import numpy as np
import numpy.typing as npt

Tensor = npt.NDArray[np.float64]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Embedding:
    pass

class SoftmaxCrossEntropy:
    def __init__(self):
        input_cache: Tensor | None = None
        
    def forward(self, x: Tensor) -> Tensor:
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
        
        self.W_y:   Tensor = np.random.randn(hidden_dim, output_dim)
        self.b_y:   Tensor = np.zeros((output_dim))
        
        self.input_cache: Tensor | None = None
    
    # f_t = sigmoid(W_f * [h_t-1, x_t] + bf) # What to keep from previous cell state
    # i_t = sigmoid(W_i * [h_t-1, x_t] + bo) # What to keep from candidate cell state
    # can_t = tanh(W_can * [h_t-1, x_t] + bcan) # What is candidate cell state
    # o_t = sigmoid(W_o * [h_t-1, x_t] + bo) # What to put into hidden state from cell state
    
    # c_t = f_t * c_t-1 + i_t * can_t # What is remaining from previous cell state + what comes from candidate cell state
    # h_t = o_t * tanh(c_t) # What to put as hidden state from cell_state
    # h_t is like a filtered output - just a part of whats on the conveyor belt c_t
    
    
    
    def forward(self, x: Tensor, init_states: tuple[Tensor, Tensor]) -> Tensor: # init_states[0] is c_prev, [1] is h_prev
        # We init both h_prev and c_prev to not lobotomize the model every batch when f.i. when doing NLP
        # We assume x is (B, seq_len, input_dim)
        self.input_cache = x
        
        B, seq_len, _ = x.shape
        
        self.batch_size = B
        self.seq_len = seq_len
        
        if init_states is None:
            c_prev = np.zeros((B, self.hidden_dim))
            h_prev = np.zeros((B, self.hidden_dim))
        else:
            c_prev, h_prev = init_states
            
        self.init_c_prev = c_prev
        self.init_h_prev = h_prev
        
        self.cell_states = []
        self.hidden_states = []
        
        self.f_gates = []
        self.i_gates = []
        self.can_states = []
        self.o_gates = []
            
        for t in range(seq_len):
            x_t = x[:, t, :] # (B, input_dim)
            
            combined_input = np.concatenate([h_prev, x_t], axis=1) # (B, hidden_dim + input_dim)
            
            f_t = sigmoid(combined_input @ self.W_f + self.b_f)   # (B, hidden_dim + input_dim) @ (hidden_dim + input_dim, hidden_dim) + (hidden_dim,) -> (B, hidden_dim)
            i_t = sigmoid(combined_input @ self.W_i + self.b_i)   # (B, hidden_dim + input_dim) @ (hidden_dim + input_dim, hidden_dim) + (hidden_dim,) -> (B, hidden_dim)
            can_t = np.tanh(combined_input @ self.W_can + self.b_can) # (B, hidden_dim + input_dim) @ (hidden_dim + input_dim, hidden_dim) + (hidden_dim,) -> (B, hidden_dim)
            o_t = sigmoid(combined_input @ self.W_o + self.b_o)   # (B, hidden_dim + input_dim) @ (hidden_dim + input_dim, hidden_dim) + (hidden_dim,) -> (B, hidden_dim)
            
            c_t = f_t * c_prev + i_t * can_t # element-wise (Hadamard product) (B, hidden_dim)
            h_t = o_t * np.tanh(c_t) # What to put into hidden (short term memory) (B, hidden_dim)
            
            self.cell_states.append(c_t)
            self.hidden_states.append(h_t)
            
            self.f_gates.append(f_t)
            self.i_gates.append(i_t)
            self.can_states.append(can_t)
            self.o_gates.append(o_t)
            
            c_prev = c_t
            h_prev = h_t
        
        self.cell_states = np.stack(self.cell_states, axis=1) # (B, seq_len, hidden_dim)
        self.hidden_states = np.stack(self.hidden_states, axis=1) # (B, seq_len, hidden_dim)
        
        self.f_gates = np.stack(self.f_gates, axis=1)
        self.i_gates = np.stack(self.i_gates, axis=1)
        self.can_states = np.stack(self.can_states, axis=1)
        self.o_gates = np.stack(self.o_gates, axis=1)
        
        output = self.hidden_states @ self.W_y + self.b_y # (B, seq_len, hidden_dim) @ (hidden_dim, output_dim) + (output_dim,)
        # We make predictions out of data that was exposed from cell states (h's)
        
        return output, c_t, h_t # Return predictions + last states
        
    
    def backward(self, dlogits: Tensor) -> Tensor:
        # Returns grad wrt. inputs
        self.dW_f = np.zeros((self.input_dim + self.hidden_dim, self.hidden_dim))
        self.dW_i = np.zeros((self.input_dim + self.hidden_dim, self.hidden_dim))
        self.dW_can = np.zeros((self.input_dim + self.hidden_dim, self.hidden_dim))
        self.dW_o = np.zeros((self.input_dim + self.hidden_dim, self.hidden_dim))
        
        self.db_f = np.zeros((self.hidden_dim))
        self.db_i = np.zeros((self.hidden_dim))
        self.db_can = np.zeros((self.hidden_dim))
        self.db_o = np.zeros((self.hidden_dim))
        
        dhidden_states = dlogits @ self.W_y.T # (B, seq_len, output_dim) @ (output_dim, hidden_dim)
        flat_dlogits = dlogits.reshape(-1, self.output_dim) # (B * seq_len, output_dim)
        flat_hidden = self.hidden_states.reshape(-1, self.hidden_dim) # (B * seq_len, hidden_dim)
        
        self.dW_y = flat_hidden.T @ flat_dlogits # (hidden_dim, B * seq_len) @ (B * seq_len, output_dim)
        self.db_y = np.sum(dlogits, axis=(0, 1))
        
        dc_next = np.zeros((self.batch_size, self.hidden_dim))
        dh_next = np.zeros((self.batch_size, self.hidden_dim))
        
        for t in range(self.seq_len - 1, -1, -1):
            c_t = self.cell_states[:, t, :] # (B, hidden_dim)
            h_t = self.hidden_states[:, t, :]
            
            c_prev_t = self.cell_states[:, t-1, :] if t != 0 else self.init_c_prev
            h_prev_t = self.hidden_states[:, t-1, :] if t != 0 else self.init_h_prev
            
            f_t = self.f_gates[:, t, :]
            i_t = self.i_gates[:, t, :]
            can_t = self.can_states[:, t, :]
            o_t = self.o_gates[:, t, :]
            
            dh_t = dhidden_states[:, t, :] # (B, hidden_dim)
            dh_total_t = dh_t + dh_next # Used in cell state grad calc. Captures how h_t influences t, t+1...
            
            do_t = dh_total_t * np.tanh(c_t) # dL/do_t = dL/dh_t * dh_t/do_t
            
            dact_c_t = dh_total_t * o_t
            dc_t = (1 - np.tanh(c_t) ** 2) * dact_c_t # dL/dc_t = dL/dact_c * dact_c / dc_t
            dc_total_t = dc_t + dc_next
            
            df_t = dc_total_t * c_prev_t # dL/df_t = dL/dc_t * dc_t/df_t
            dc_prev_t = dc_total_t * f_t # dL/dc_prev_t = dL/dc_t * dc_t/dc_prev_t
            di_t = dc_total_t * can_t # dL/di_t = dL/dc_t * dc_t/di_t
            dcan_t = dc_total_t * i_t # dL/dcan_t = dL/dc_t * dc_t/dcan_t
            
            # Derivative of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            
            
            
            
            
            
            
            
            
         
        
        
        
        pass
    
    def step(self):
        pass