import numpy as np
import numpy.typing as npt

Tensor = npt.NDArray[np.float64]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class CosineScheduler:
    def __init__(self, max_lr, min_lr, warmup_epochs, total_epochs):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def get_lr(self, epoch):
        if epoch <= self.warmup_epochs:
            return self.max_lr * (epoch / self.warmup_epochs)
        
        else:
            total_decay_epochs = self.total_epochs - epoch
            current_decay_epoch = epoch - self.warmup_epochs

            coeff = current_decay_epoch / total_decay_epochs

            return self.min_lr + (1 / 2) * (self.max_lr - self.min_lr) * (1 + np.pi * coeff)


class Embedding:
    def __init__(self, input_dim, embed_dim):
        self.input_cache: Tensor | None = None
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.embeddings = np.random.randn(input_dim, embed_dim)
        
    def forward(self, x):
        self.input_cache = x
        
        return self.embeddings[x]
    
    def backward(self, out_grad):
        self.dembeddings = np.zeros((self.input_dim, self.embed_dim))
        np.add.at(self.dembeddings, self.input_cache, out_grad)
        
        return self.dembeddings
    
    def step(self, learning_rate, clip_val=0.0):
        if clip_val != 0.0:
            np.clip(self.dembeddings, -clip_val, clip_val, self.dembeddings)

        self.embeddings -= learning_rate * self.dembeddings
        
class SoftmaxCrossEntropy:
    def __init__(self):
        self.logits: Tensor | None = None
        self.targets: Tensor | None = None
        self.probs: Tensor | None = None
        
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        self.logits = logits # (B, seq_len, output_dim)
        self.targets = targets

        max_logits = np.max(logits, axis=2, keepdims=True) # (B, seq_len, 1)
        shifted_logits = logits - max_logits # (B, seq_len, output_dim) - (B, seq_len, 1)
        
        exp_logits = np.exp(shifted_logits)
        exp_sum = np.sum(exp_logits, axis=2, keepdims=True) # (B, seq_len, 1)
        probs = exp_logits / exp_sum 
        self.probs = probs
        
        log_probs = np.log(probs)
        log_probs = log_probs * targets # targets is one-hot encoded
        batch_loss = -np.sum(log_probs, axis=-1) # (B, seq_len)
        batch_loss = np.mean(batch_loss) # Scalar
        
        return batch_loss
    
    def backward(self):
        batch_size = self.logits.shape[0]
        dlogits = (self.probs - self.targets) / batch_size # bc we did np.mean
        return dlogits
        
    
    

class MSELoss:
    def __init__(self):
        self.preds: Tensor | None = None
        self.targets: Tensor | None = None
    
    def forward(self, preds, targets):
        self.preds = preds
        self.targets = targets
        
        loss = (targets - preds) ** 2 # we need same shapes
        return np.mean(loss)
    
    def backward(self):
        grad = 2 / self.preds.shape[0] * (self.preds - self.targets)
        
        return grad
    
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

    def step(self):
        self.t += 1

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
            v_hat = self.m[i] / (1 - self.beta2 ** self.t) # same story

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps) # accounts for direction (m_hat) and intensity (v_hat)
            # The n_hat works as momentum of direction, and v_hat as scaling if gradients are huge or tiny.

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
        
    def params(self):
        return [
            (self.W_f,   self.dW_f),
            (self.W_i,   self.dW_i),
            (self.W_can, self.dW_can),
            (self.W_o,   self.dW_o),
            
            (self.b_f,   self.db_f),
            (self.b_i,   self.db_i),
            (self.b_can, self.db_can),
            (self.b_o,   self.db_o),
            
            (self.W_y,   self.dW_y),
            (self.b_y,   self.db_y)
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
        
        dx_ts = []
        
        for t in range(self.seq_len - 1, -1, -1):
            x_t = self.input_cache[:, t, :]
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
            dc_total_t = dc_t + dc_next # How cell state at t influences t, t_1, ...
            
            df_t = dc_total_t * c_prev_t # dL/df_t = dL/dc_t * dc_t/df_t
            dc_prev_t = dc_total_t * f_t # dL/dc_prev_t = dL/dc_t * dc_t/dc_prev_t
            di_t = dc_total_t * can_t # dL/di_t = dL/dc_t * dc_t/di_t
            dcan_t = dc_total_t * i_t # dL/dcan_t = dL/dc_t * dc_t/dcan_t
            
            combined_input_t = np.concatenate([h_prev_t, x_t], axis=1) # (B, hidden_dim + input_dim)
            
            # Derivative of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            df_preact_t = df_t * f_t * (1 - f_t) # dL/dpreact = dL/dact * dact/dpreact
            di_preact_t = di_t * i_t * (1 - i_t)
            dcan_preact_t = dcan_t * (1 - can_t ** 2) # tanh instead of sigmoid
            do_preact_t = do_t * o_t * (1 - o_t)
            
            # dL/dW = X.T @ dL/dh
            # dL/dX = dL/dh @ W.T
            
            # The preacts are our dL/dh
            dcomb_inp_t = np.zeros((self.batch_size, self.input_dim + self.hidden_dim))
            
            dcomb_inp_t += df_preact_t @ self.W_f.T # (B, hidden_dim) @ (hidden_dim, hidden_dim + input_dim) = (B, hidden_dim + input_dim)
            dcomb_inp_t += di_preact_t @ self.W_i.T # (B, hidden_dim) @ (hidden_dim, hidden_dim + input_dim) = (B, hidden_dim + input_dim)
            dcomb_inp_t += dcan_preact_t @ self.W_can.T # (B, hidden_dim) @ (hidden_dim, hidden_dim + input_dim) = (B, hidden_dim + input_dim)
            dcomb_inp_t += do_preact_t @ self.W_o.T # (B, hidden_dim) @ (hidden_dim, hidden_dim + input_dim) = (B, hidden_dim + input_dim)
            
            self.dW_f += combined_input_t.T @ df_preact_t # (hidden_dim + input_dim, B) @ (B, hidden_dim) = (hidden_dim + input_dim, hidden_dim)
            self.dW_i += combined_input_t.T @ di_preact_t # (hidden_dim + input_dim, B) @ (B, hidden_dim) = (hidden_dim + input_dim, hidden_dim)
            self.dW_can += combined_input_t.T @ dcan_preact_t # (hidden_dim + input_dim, B) @ (B, hidden_dim) = (hidden_dim + input_dim, hidden_dim)
            self.dW_o += combined_input_t.T @ do_preact_t # (hidden_dim + input_dim, B) @ (B, hidden_dim) = (hidden_dim + input_dim, hidden_dim)
            
            # input @ W + b = (B, hidden_dim)
            self.db_f += np.sum(df_preact_t, axis=0) # (hidden_dim,)
            self.db_i += np.sum(di_preact_t, axis=0) # (hidden_dim,)
            self.db_can += np.sum(dcan_preact_t, axis=0) # (hidden_dim,)
            self.db_o += np.sum(do_preact_t, axis=0) # (hidden_dim,)
            
            # We concatenated the combined_input as concat [h_prev, x_t] therefore h_prev is [:, h_prev:, :] and x_t is rest
            
            dh_prev_t = dcomb_inp_t[:, :self.hidden_dim] # (B, hidden_dim)
            dx_t = dcomb_inp_t[:, self.hidden_dim:] # (B, input_dim)
            
            dx_ts.append(dx_t)

            dh_next = dh_prev_t
            dc_next = dc_prev_t
            
            np.clip(dh_next, -1, 1, dh_next)
            np.clip(dc_next, -1, 1, dc_next)
            
        return np.stack(dx_ts[::-1], axis=1) # Reverse back, (B, seq_len, input_dim)
    
    
    def step(self, learning_rate: float, clip_val: float = 1.0):
        for p, grad in self.params():
            if clip_val != 0.0:
                np.clip(grad, -clip_val, clip_val, grad)
        
        self.W_f -= learning_rate * self.dW_f
        self.W_i -= learning_rate * self.dW_i
        self.W_can -= learning_rate * self.dW_can
        self.W_o -= learning_rate * self.W_o
        
        self.b_f -= learning_rate * self.db_f
        self.b_i -= learning_rate * self.db_i
        self.b_can -= learning_rate * self.db_can
        self.b_o -= learning_rate * self.db_o
        
        self.W_y -= learning_rate * self.dW_y
        self.b_y -= learning_rate * self.db_y
        
        