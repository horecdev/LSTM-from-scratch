import cupy as cp
import numpy.typing as npt

from src.core.layers import sigmoid

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
        grad = 2 / (self.preds.size) * (self.preds - self.targets)
        
        return grad
    
class SigmoidWeightedBCELoss:
    def __init__(self):
        self.preds: Tensor | None = None
        self.targets: Tensor | None = None
        self.weights: Tensor | None = None
        
    # Just like SoftmaxCrossEntropy, BCELoss is more stable with Sigmoid activation (no blowing into NaNs)
        
    # The formula for BCE is -[y * log(p) + (1 - y)log(1 - p)]
    # If target is 1 then formula becomes -log(p) so it is 0 when p is 1. When p - 0.01 then loss is like 4.6
    # If target is 0 then formula becomes -log(1 - p) so it is 0 when p = 0 and a lot when p = 0.99
    # It is more aggresive than MSE because in MSE max loss per sample is 1.0 (for masks)
        
    def forward(self, logits: Tensor, targets: Tensor, weights: Tensor) -> tuple[float, Tensor]:
        # Do sigmoid
        self.preds = sigmoid(logits)
        self.preds = cp.clip(self.preds, 1e-7, 1 - 1e-7)
        self.targets = targets
        self.weights = weights

        loss_val = -(targets * cp.log(self.preds) + (1 - targets) * cp.log(1 - self.preds))
        loss = cp.mean(loss_val * self.weights)

        return loss, self.preds
    
    def backward(self):
        # This is grad wrt. logits, the derivative of lsos wrt. preds (just BCE without sigmoid) is (p - y) / (p(1-p)) but the p's
        # are activated by sigmoid and the derivative of loss wrt. logits is dL/dpreds * dpreds/dlogits and dpreds/dlogits is p(1-p) so it simplifies to p - y.
        return (self.weights * (self.preds - self.targets)) / self.preds.size # multiplying by weights decreases grads of non important samples