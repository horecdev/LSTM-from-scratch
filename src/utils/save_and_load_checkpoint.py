import cupy as cp

def save_checkpoint(file_path, layers_dict, optimizer, epoch, best_loss):
    data_to_save = {}
    
    for layer_name, layer_obj in layers_dict.items():
        for i, (p, _) in enumerate(layer_obj.params()):
            data_to_save[f"{layer_name}_{i}"] = p
            
    adam_state = optimizer.get_state()
    
    for i, m_tensor in enumerate(adam_state['m']):
        data_to_save[f"adam_m_{i}"] = m_tensor
    for i, v_tensor in enumerate(adam_state['v']):
        data_to_save[f"adam_v_{i}"] = v_tensor
    data_to_save["adam_t"] = cp.array(adam_state['t']) # Save as 1 elem array
    data_to_save["epoch"] = cp.array(epoch)
    data_to_save["best_loss"] = cp.array(best_loss)
    
    cp.savez(file_path, **data_to_save)
    
    print(f"Saved checkpoint to: {file_path}")
    
def load_checkpoint(file_path, layers_dict, optimizer):
    data = cp.load(file_path)
    
    for layer_name, layer_obj in layers_dict.items():
        for i, (p, _) in enumerate(layer_obj.params()):
            cp.copyto(p, data[f"{layer_name}_{i}"])
    
    m_list = []
    v_list = []            
    i = 0
    while f"adam_m_{i}" in data: # They have smae amount of elements so it doesnt matter whether m or v
        m_list.append(data[f"adam_m_{i}"])
        v_list.append(data[f"adam_v_{i}"])
        i += 1
        
    optimizer.load_state({
        "m" : m_list,
        "v" : v_list,
        "t" : int(data['adam_t'][0]) # get the first elem
    })
    
    start_epoch = int(data["epoch"][0])
    best_loss = int(data['best_loss'][0])
    print(f"Loaded checkpoint from {file_path} at epoch {start_epoch}")
    return start_epoch, best_loss
    
        
        
