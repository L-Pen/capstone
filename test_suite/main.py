from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from helper import save_results_to_file,generate_params,load_data
from model_test import run_model_test
import time

param_options = {
    'group_size':25,
    'TEST_PROPORTION':0.2,
    'epochs':1000,
    'batch_size':256,
    'output_range':[[-1,1]],
    'between_layers':[2,3],
    'first_layer_size':[16,32,64,128],
    'layer_size_factor':[0.5,0.75],
    'bottleneck_size':2,
    'num_features':26,
    'bottleneck_activation':['linear','tanh'],
    'output_activation':['linear','tanh'],
}

def run_test(params):
    startTime = time.time()
    result = run_model_test(params)
    total_time = time.time() - startTime
    print(f'finished {params} in {total_time} seconds with result {result}')
    return (result,params,total_time)

if __name__ == "__main__":
    params = generate_params(param_options)
    #add id to each param
    for i in range(len(params)):
        params[i]['id'] = i
    #save params to json file

    pd.DataFrame(params).to_json("params.json",orient="records")
    print("cores available:",cpu_count())
    print("number of tests:",len(params))
    num_processes= 1
    if cpu_count() > 16:
        num_processes = 6
    print("number of processes:",num_processes)
    with Pool(processes=1) as pool:
        results = pool.map(run_test,params)
    save_results_to_file(results)
 
