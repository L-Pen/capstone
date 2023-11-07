from multiprocessing import Pool, cpu_count
import numpy as np
from helper import save_results_to_file,generate_params
from time import sleep

param_options = {
    "A": [1,2,3],
    "B": [4,5,6],
    "C": [7,8,9],
}

def run_test(params):
    result = np.random.rand()
    sleep(1)
    print((result,params))
    return (result,params)

if __name__ == "__main__":
    params = generate_params(param_options)
    print("cores available:",cpu_count())
    print("number of tests:",len(params))
    with Pool() as pool:
        results = pool.map(run_test, params)
    results.sort(key=lambda x: x[0])
    save_results_to_file(results)
    print("best result: ",results[-1])
