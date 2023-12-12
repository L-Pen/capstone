from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from helper import save_results_to_file,generate_params,load_data
from model_test import run_model_test
import time

param_options = {
    'group_size':[25],
    'TEST_PROPORTION':0.25,
    'epochs':1000,
    'batch_size':[256],
    'model':{
        'layers':[
            {'type':'input','shape':(26,)},
            {'type':'dense','units':32,'activation':'relu'},
            {'type':'dense','units':16,'activation':'elu'},
            {'type':'dense','units':3,'activation':'linear','name':'Bottleneck'}, # The bottleneck. 
            {'type':'dense','units':16,'activation':'elu'},
            {'type':'dense','units':32,'activation':'relu'},
            {'type':'dense','units':26,'activation':'linear'},
        ],
        'output_range':[-1,1]
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
    #add id to each param
    for i in range(len(params)):
        params[i]['id'] = i
    #save params to json file

    pd.DataFrame(params).to_json("params.json",orient="records")
    print("cores available:",cpu_count())
    print("number of tests:",len(params))
    with Pool(processes=1) as pool:
        results = pool.map(run_test,params)
    save_results_to_file(results)
 
