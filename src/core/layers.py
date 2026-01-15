import numpy as np
import numpy.typing as npt

Tensor = npt.NDArray[np.float64]

class SigmoidActivation:
    def __init__(self):
        self.output_cache: Tensor | None = None
        
    def forward(self, x: Tensor) -> Tensor:
        output = 1 / (1 + np.exp(-x))
        self.output_cache = output
        return output
        
    def backward(self, out_grad: Tensor) -> Tensor:
        grad = self.output_cache * (1 - self.output_cache) * out_grad
        return grad
    
def sigmoid(x: Tensor) -> Tensor: # For activation in LSTM
    return 1 / (1 + np.exp(-x))


class Embedding:
    def __init__(self, input_dim, embed_dim):
        self.input_cache: Tensor | None = None
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.embeddings = np.random.randn(input_dim, embed_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        self.input_cache = x
        
        return self.embeddings[x]
    
    def backward(self, out_grad: Tensor) -> Tensor:
        self.dembeddings = np.zeros((self.input_dim, self.embed_dim))
        np.add.at(self.dembeddings, self.input_cache, out_grad)
        
        return self.dembeddings
    
    def step(self, learning_rate, clip_val=0.0):
        if clip_val != 0.0:
            np.clip(self.dembeddings, -clip_val, clip_val, self.dembeddings)

        self.embeddings -= learning_rate * self.dembeddings
        

class Linear: # Works only as the LSTM head, not normal (B, dim) projection.
    def __init__(self, input_dim, output_dim):
        self.input_cache: Tensor | None = None
        
        self.W: Tensor = np.random.randn(input_dim, output_dim)
        self.b: Tensor = np.zeros(output_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        self.input_cache = x
        return x @ self.W + self.b
    
    def backward(self, out_grad: Tensor) -> Tensor: # We have to flatten to calc accurately. Otherwise we will have shape errors
        # This trick works because dot product accumulates the gradient between sampels for us. Same with db and .sum()
        # Intuition: Math works out without issues when input is (B, dim) so we flatten it into (B * seq_len, dim)
        B, seq_len, input_dim = self.input_cache.shape
        _, _, output_dim = out_grad.shape
        
        x_flat = self.input_cache.reshape(-1, input_dim) # (B * seq_len, input_dim)
        grad_flat = out_grad.reshape(-1, output_dim) # (B * seq_len, output_dim)
        
        self.dW = x_flat.T @ grad_flat # (input_dim, B * seq_len) @ (B * seq_len, output_dim) = (input_dim, output_dim)
        self.db = np.sum(out_grad, axis=0)
        
        dx_flat = grad_flat @ self.W.T # (B * seq_len, output_dim) @ (output_dim, input_dim) = (B * seq_len, input_dim)
        
        return dx_flat.reshape(B, seq_len, input_dim) # Gradient wrt. input
    
    def params(self):
        return [(self.W, self.dW), {self.b, self.db}]