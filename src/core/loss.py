import cupy as cp
import numpy.typing as npt

Tensor = npt.NDArray[cp.float64]

class SoftmaxCrossEntropy:
    def __init__(self):
        self.logits: Tensor | None = None
        self.targets: Tensor | None = None
        self.probs: Tensor | None = None
        
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        self.logits = logits # (B, seq_len, output_dim)
        self.targets = targets

        max_logits = cp.max(logits, axis=2, keepdims=True) # (B, seq_len, 1)
        shifted_logits = logits - max_logits # (B, seq_len, output_dim) - (B, seq_len, 1)
        
        exp_logits = cp.exp(shifted_logits)
        exp_sum = cp.sum(exp_logits, axis=2, keepdims=True) # (B, seq_len, 1)
        probs = exp_logits / exp_sum 
        self.probs = probs
        
        log_probs = cp.log(probs)
        log_probs = log_probs * targets # targets is one-hot encoded
        batch_loss = -cp.sum(log_probs, axis=-1) # (B, seq_len)
        batch_loss = cp.mean(batch_loss) # Scalar
        
        return batch_loss
    
    def backward(self) -> Tensor:
        batch_size = self.logits.shape[0]
        dlogits = (self.probs - self.targets) / batch_size # bc we did cp.mean
        return dlogits
        
    
class MSELoss:
    def __init__(self):
        self.preds: Tensor | None = None
        self.targets: Tensor | None = None
    
    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        self.preds = preds
        self.targets = targets
        
        loss = (targets - preds) ** 2 # we need same shapes
        return cp.mean(loss)
    
    def backward(self) -> Tensor:
        grad = 2 / self.preds.shape[0] * (self.preds - self.targets)
        
        return grad