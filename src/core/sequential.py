import cupy as cp

class Sequential:
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad # return the last grad wrt. input
    
    def params(self):
        all_params = []
        
        for layer in self.layers:
            if hasattr(layer, 'params'):
                all_params.extend(layer.params())
                
        return all_params
    
    def summary(self):
        param_count = 0
        
        for p_tuple in self.params():
            for p, _ in p_tuple:
                param_count += p.size
                
        print(f"Total trainable parameters: {param_count}")