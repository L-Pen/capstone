from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from helper import save_results_to_file,generate_params,load_data
from model_test import run_model_test

from tensorflow.keras import layers, losses
import time

param_options = {
    'group_size':[15,20,25],
    'TEST_PROPORTION':0.25,
    'epochs':1000,
    'batch_size':[1024,512,256],
    'model':{
        'layers':[
            layers.InputLayer(input_shape=(26,)),
            layers.Dense(13, activation='elu'),
            layers.Dense(7, activation='relu'),
            layers.Dense(3, activation='linear', name="Bottleneck"), # The bottleneck. 
            layers.Dense(7, activation='elu'),
            layers.Dense(13, activation='relu'),
            layers.Dense(26, activation='linear'),
        ],
        'optimizer':'adagrad',
        'loss':losses.MeanSquaredError(),
        'metrics':['mse'],
    }
}

def run_test(params):
    startTime = time.time()
    result = run_model_test(params)
    total_time = time.time() - startTime
    print(f'finished {params} in {total_time} seconds with result {result}')
    return (result,params,total_time)

if __name__ == "__main__":
    params = generate_params(param_options)
    print("cores available:",cpu_count())
    print("number of tests:",len(params))
    with Pool(processes=6) as pool:
        results = pool.map(run_test,params)
    save_results_to_file(results)
 
