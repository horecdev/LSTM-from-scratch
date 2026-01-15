import cupy as cp

def save_model(file_path, layer_dict):
    # We want to go over each layer and save its weights
    # We assume layer dict is {name : model}
    
    weights_to_save = {}
    
    for layer_name, layer_obj in layer_dict.items():
        params = layer_obj.params()
        
        
        for i, (p, _) in enumerate(params):
            weights_to_save[f"{layer_name}_{i}"] = p
            
        cp.savez(file_path, **weights_to_save)
        print(f"Model saved to {file_path}")
        
def load_model(file_path, layer_dict):
    # We load back the params in-place to variables from params
    # The layer dict can be any amount of models that have params() func
    data = cp.load(file_path)
    
    for layer_name, layer_obj in layer_dict.items():
        params = layer_obj.params()
        
        for i, (p, _) in enumerate(params):
            key = f"{layer_name}_{i}"
            
            if key in data:
                cp.copyto(p, data[key])
            else:
                print(f"Warning: Layer {key} not found in saved file.")
                
    print(f"Model loaded from {file_path}")
            