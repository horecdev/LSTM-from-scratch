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
    data = cp.load(file_path)
    keys = data.npz_file.files
    
    for layer_name, layer_obj in layer_dict.items():
        # Get params
        params = layer_obj.params() 
        
        for i, (p, _) in enumerate(params):
            key = f"{layer_name}_{i}" # Matches the save_checkpoint format
            if key in keys:
                cp.copyto(p, data[key])
                
    print(f"Successfully loaded model from {file_path}")
            