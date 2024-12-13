import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

def open_json(file_name):
    # Read JSON file
    with open(file_name, 'r') as file:
        content = json.load(file)

    # Flatten lists and convert to dataframe
    dataset = [item for sublist in content.values() for item in sublist]
    data = pd.DataFrame(dataset)
    return data
    

def split_data(data, ratio, random_state):
    train_data, test_data = train_test_split(data, test_size=(1 - ratio), random_state=random_state)
    return train_data, test_data


def get_bootstrap_samples(data, ratio, number_bootstrap, seed):

    bootstrap = []  
    test = []  

    np.random.seed(seed)

    for i in range(number_bootstrap):
        sample = data.sample(frac=ratio, replace=True, random_state=i)  
        out_of_sample_data = data.loc[data.index.difference(sample.index)]  
        bootstrap.append(sample.reset_index(drop=True))  
        test.append(out_of_sample_data.reset_index(drop=True))  
    
    return bootstrap, test
    
    