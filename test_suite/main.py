from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from helper import save_results_to_file,generate_params,load_data
import time
param_options = {
    "A": [1,2,3],
    "B":[4,5]
}

def run_test(params):
    X,y = load_data(10)
    result = np.random.rand()
    return (result,params)

if __name__ == "__main__":
    params = generate_params(param_options)
    print("cores available:",cpu_count())
    print("number of tests:",len(params))
    times = []
    for i in range(10):
        startTime = time.time()
        with Pool(processes=i+1) as pool:
            results = pool.map(run_test,params)
        endTime = time.time()
        print(f'finished {i+1} processes in {endTime-startTime} seconds')
        times.append(endTime-startTime)
    print(times)
