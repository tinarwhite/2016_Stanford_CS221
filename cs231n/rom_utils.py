import os
import numpy as np

def load_burg_data(*args):
    data = {}
    base_dir='cs231n/datasets/burg'
    data['features'] = np.vstack([[float(i) for i in np.hstack([x.replace('p', '.').split('_')[1:3], 
                                                                x.replace('p', '.').split('_')[3].split('.dat')[0]])] for x in args])
    data['solutions'] = np.swapaxes(np.dstack([np.loadtxt(os.path.join(base_dir, x)) for x in args]),0,2)
    return data

def sample_burg_minibatch(data, batch_size=1):
    split_size = data['solutions'].shape[0]
    mask = np.random.choice(split_size, batch_size)
    solutions = data['solutions'][mask]
    features = data['features'][mask]
    return solutions, features