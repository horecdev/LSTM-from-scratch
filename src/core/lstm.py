import cupy as cp
import numpy.typing as npt

from src.core.layers import sigmoid

Tensor = npt.NDArray[cp.float64]
class LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Rule of thumb for initialization:
        # If Sigmoid / Tanh: Uniform(-limit, limit) where limit is sqrt(6 / (fan_in + fan_out))
        # If ReLU: Normal(mean=0, std=sqrt(2/fan_in))
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        fan_in = input_dim + hidden_dim
        fan_out = hidden_dim
        limit = cp.sqrt(6 / (fan_in + fan_out))
        
        self.W_f:   Tensor = cp.random.uniform(-limit, limit, (fan_in, fan_out))
        self.W_i:   Tensor = cp.random.uniform(-limit, limit, (fan_in, fan_out))
        self.W_can: Tensor = cp.random.uniform(-limit, limit, (fan_in, fan_out))
        self.W_o:   Tensor = cp.random.uniform(-limit, limit, (fan_in, fan_out))
        
        self.b_f:   Tensor = cp.ones((hidden_dim)) # We make the gates forget less info at the start of training
        self.b_i:   Tensor = cp.zeros((hidden_dim))
        self.b_can: Tensor = cp.zeros((hidden_dim))
        self.b_o :  Tensor = cp.zeros((hidden_dim))
        
        self.input_cache: Tensor | None = None
        
        # Init grads for optimizer
        self.dW_f = cp.zeros((self.input_dim + self.hidden_dim, self.hidden_dim))
        self.dW_i = cp.zeros((self.input_dim + self.hidden_dim, self.hidden_dim))
        self.dW_can = cp.zeros((self.input_dim + self.hidden_dim, self.hidden_dim))
        self.dW_o = cp.zeros((self.input_dim + self.hidden_dim, self.hidden_dim))
        
        self.db_f = cp.zeros((self.hidden_dim))
        self.db_i = cp.zeros((self.hidden_dim))
        self.db_can = cp.zeros((self.hidden_dim))
        self.db_o = cp.zeros((self.hidden_dim))
        
    def params(self):
        return [
            (self.W_f,   self.dW_f),
            (self.W_i,   self.dW_i),
            (self.W_can, self.dW_can),
            (self.W_o,   self.dW_o),
            
            (self.b_f,   self.db_f),
            (self.b_i,   self.db_i),
            (self.b_can, self.db_can),
            (self.b_o,   self.db_o)
        ]
    
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
            c_prev = cp.zeros((B, self.hidden_dim))
            h_prev = cp.zeros((B, self.hidden_dim))
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
            
            combined_input = cp.concatenate([h_prev, x_t], axis=1) # (B, hidden_dim + input_dim)
            
            f_t = sigmoid(combined_input @ self.W_f + self.b_f)   # (B, hidden_dim + input_dim) @ (hidden_dim + input_dim, hidden_dim) + (hidden_dim,) -> (B, hidden_dim)
            i_t = sigmoid(combined_input @ self.W_i + self.b_i)   # (B, hidden_dim + input_dim) @ (hidden_dim + input_dim, hidden_dim) + (hidden_dim,) -> (B, hidden_dim)
            can_t = cp.tanh(combined_input @ self.W_can + self.b_can) # (B, hidden_dim + input_dim) @ (hidden_dim + input_dim, hidden_dim) + (hidden_dim,) -> (B, hidden_dim)
            o_t = sigmoid(combined_input @ self.W_o + self.b_o)   # (B, hidden_dim + input_dim) @ (hidden_dim + input_dim, hidden_dim) + (hidden_dim,) -> (B, hidden_dim)
            
            c_t = f_t * c_prev + i_t * can_t # element-wise (Hadamard product) (B, hidden_dim)
            h_t = o_t * cp.tanh(c_t) # What to put into hidden (short term memory) (B, hidden_dim)
            
            self.cell_states.append(c_t)
            self.hidden_states.append(h_t)
            
            self.f_gates.append(f_t)
            self.i_gates.append(i_t)
            self.can_states.append(can_t)
            self.o_gates.append(o_t)
            
            c_prev = c_t
            h_prev = h_t
        
        self.cell_states = cp.stack(self.cell_states, axis=1) # (B, seq_len, hidden_dim)
        self.hidden_states = cp.stack(self.hidden_states, axis=1) # (B, seq_len, hidden_dim)
        
        self.f_gates = cp.stack(self.f_gates, axis=1)
        self.i_gates = cp.stack(self.i_gates, axis=1)
        self.can_states = cp.stack(self.can_states, axis=1)
        self.o_gates = cp.stack(self.o_gates, axis=1)
        
        return self.hidden_states, c_t, h_t # Return hidden_states to later make preds out of them. We isolate W_y into some totally different part.
        
    
    def backward(self, dhidden_states: Tensor) -> Tensor:
        # dhidden_states is grad wrt. hidden_states, shape (B, seq_len, hidden_dim)
        # Returns grad wrt. inputs
        
        dc_next = cp.zeros((self.batch_size, self.hidden_dim))
        dh_next = cp.zeros((self.batch_size, self.hidden_dim))
        
        dx_ts = []
        
        for t in range(self.seq_len - 1, -1, -1):
            x_t = self.input_cache[:, t, :]
            c_t = self.cell_states[:, t, :] # (B, hidden_dim)
            h_t = self.hidden_states[:, t, :] # Not needed bc we already have dh_t
            
            c_prev_t = self.cell_states[:, t-1, :] if t != 0 else self.init_c_prev
            h_prev_t = self.hidden_states[:, t-1, :] if t != 0 else self.init_h_prev
            
            f_t = self.f_gates[:, t, :]
            i_t = self.i_gates[:, t, :]
            can_t = self.can_states[:, t, :]
            o_t = self.o_gates[:, t, :]
            
            dh_t = dhidden_states[:, t, :] # (B, hidden_dim)
            dh_total_t = dh_t + dh_next # Used in cell state grad calc. Captures how h_t influences t, t+1...
            
            do_t = dh_total_t * cp.tanh(c_t) # dL/do_t = dL/dh_t * dh_t/do_t
            
            dact_c_t = dh_total_t * o_t
            dc_t = (1 - cp.tanh(c_t) ** 2) * dact_c_t # dL/dc_t = dL/dact_c * dact_c / dc_t
            dc_total_t = dc_t + dc_next # How cell state at t influences t, t_1, ...
            
            df_t = dc_total_t * c_prev_t # dL/df_t = dL/dc_t * dc_t/df_t
            dc_prev_t = dc_total_t * f_t # dL/dc_prev_t = dL/dc_t * dc_t/dc_prev_t
            di_t = dc_total_t * can_t # dL/di_t = dL/dc_t * dc_t/di_t
            dcan_t = dc_total_t * i_t # dL/dcan_t = dL/dc_t * dc_t/dcan_t
            
            combined_input_t = cp.concatenate([h_prev_t, x_t], axis=1) # (B, hidden_dim + input_dim)
            
            # Derivative of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            df_preact_t = df_t * f_t * (1 - f_t) # dL/dpreact = dL/dact * dact/dpreact
            di_preact_t = di_t * i_t * (1 - i_t)
            dcan_preact_t = dcan_t * (1 - can_t ** 2) # tanh instead of sigmoid
            do_preact_t = do_t * o_t * (1 - o_t)
            
            # dL/dW = X.T @ dL/dh
            # dL/dX = dL/dh @ W.T
            
            # The preacts are our dL/dh
            dcomb_inp_t = cp.zeros((self.batch_size, self.input_dim + self.hidden_dim))
            
            dcomb_inp_t += df_preact_t @ self.W_f.T # (B, hidden_dim) @ (hidden_dim, hidden_dim + input_dim) = (B, hidden_dim + input_dim)
            dcomb_inp_t += di_preact_t @ self.W_i.T # (B, hidden_dim) @ (hidden_dim, hidden_dim + input_dim) = (B, hidden_dim + input_dim)
            dcomb_inp_t += dcan_preact_t @ self.W_can.T # (B, hidden_dim) @ (hidden_dim, hidden_dim + input_dim) = (B, hidden_dim + input_dim)
            dcomb_inp_t += do_preact_t @ self.W_o.T # (B, hidden_dim) @ (hidden_dim, hidden_dim + input_dim) = (B, hidden_dim + input_dim)
            
            self.dW_f += combined_input_t.T @ df_preact_t # (hidden_dim + input_dim, B) @ (B, hidden_dim) = (hidden_dim + input_dim, hidden_dim)
            self.dW_i += combined_input_t.T @ di_preact_t # (hidden_dim + input_dim, B) @ (B, hidden_dim) = (hidden_dim + input_dim, hidden_dim)
            self.dW_can += combined_input_t.T @ dcan_preact_t # (hidden_dim + input_dim, B) @ (B, hidden_dim) = (hidden_dim + input_dim, hidden_dim)
            self.dW_o += combined_input_t.T @ do_preact_t # (hidden_dim + input_dim, B) @ (B, hidden_dim) = (hidden_dim + input_dim, hidden_dim)
            
            # input @ W + b = (B, hidden_dim)
            self.db_f += cp.sum(df_preact_t, axis=0) # (hidden_dim,)
            self.db_i += cp.sum(di_preact_t, axis=0) # (hidden_dim,)
            self.db_can += cp.sum(dcan_preact_t, axis=0) # (hidden_dim,)
            self.db_o += cp.sum(do_preact_t, axis=0) # (hidden_dim,)
            
            # We concatenated the combined_input as concat [h_prev, x_t] therefore h_prev is [:, h_prev:, :] and x_t is rest
            
            dh_prev_t = dcomb_inp_t[:, :self.hidden_dim] # (B, hidden_dim)
            dx_t = dcomb_inp_t[:, self.hidden_dim:] # (B, input_dim)
            
            dx_ts.append(dx_t)

            dh_next = dh_prev_t
            dc_next = dc_prev_t
            
            cp.clip(dh_next, -1, 1, dh_next)
            cp.clip(dc_next, -1, 1, dc_next)
            
        return cp.stack(dx_ts[::-1], axis=1) # Reverse back, (B, seq_len, input_dim)
    
    
    def step(self, learning_rate: float, clip_val: float = 1.0):
        for p, grad in self.params():
            if clip_val != 0.0:
                cp.clip(grad, -clip_val, clip_val, grad)
        
            p -= learning_rate * grad 
        
class BiLSTM:
    # Bidirectional LSTMs work based on computing two hidden states - one looking from 0 -> t, and one from t -> seq_len. We concat and make preds.
    # Having previous states makes no sense here, because we look into the future as far as possible. We have hindsight, and idk what the state would even look like going both ways.
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.forward_layer = LSTM(input_dim, hidden_dim, output_dim)
        self.backward_layer = LSTM(input_dim, hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        
    def forward(self, x: Tensor, init_f_states=None, init_b_states=None) -> Tensor:
        h_f, _, _ = self.forward_layer.forward(x, init_f_states)
        
        x_rev = cp.flip(x, axis=1) # flip along t axis
        h_b, _, _ = self.backward_layer.forward(x_rev, init_b_states) # Now first hidden state attends to last x, etc.
        
        h_b_rev = cp.flip(h_b, axis=1) # now first elements is last in backward lstm, this means attends from seq_len to 0
        
        combined_h = cp.concatenate([h_f, h_b_rev], axis=2) # (B, seq_len, hidden_dim * 2)
        
        return combined_h
    
    def backward(self, dhidden_states):
        dh_f = dhidden_states[:, :, :self.hidden_dim]
        dh_b_rev = dhidden_states[:, :, self.hidden_dim:]
        
        dh_b = cp.flip(dh_b_rev, axis=1)
        
        dx_f = self.forward_layer.backward(dh_f)
        dx_b_rev = self.backward_layer.backward(dh_b)
        dx_b = cp.flip(dx_b_rev, axis=1)
        
        return dx_f + dx_b # grad wrt. inputs
    
    def params(self):
        return self.forward_layer.params() + self.backward_layer.params()