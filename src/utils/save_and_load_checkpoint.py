import numpy as np

def save_checkpoint(file_path, layers_dict, optimizer):
    data_to_save = {}
    
    for layer_name, layer_obj in layers_dict:
        for i, (p, _) in enumerate(layer_obj.params()):
            data_to_save[f"{layer_name}_{i}"] = p
            
    adam_state = optimizer.get_state()
    
    for i, m_tensor in enumerate(adam_state['m']):
        data_to_save[f"adam_m_{i}"] = m_tensor
    for i, v_tensor in enumerate(adam_state['v']):
        data_to_save[f"adam_m_{i}"] = v_tensor
    data_to_save["adam_t"] = np.array(adam_state['t']) # Save as 1 elem array
    
    np.savez(file_path, data_to_save)
    
def load_checkpoint(file_path, layers_dict, optimizer):
    data = np.load(file_path)
    
    for layer_name, layer_obj in layers_dict:
        for i, (p, _) in enumerate(layer_obj.params()):
            np.copyto(p, data[f"{layer_name}_{i}"])
    
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
    
    print(f"Loaded checkpoint from {file_path}.")
    
        
        
