from multiprocessing import Pool
import numpy as np
from helper import save_results_to_file,generate_params

param_options = {
    "A": [1,2,3],
    "B": [4,5,6],
}

def run_test(params):
    result = np.random.rand()
    print((result,params))
    return (result,params)

if __name__ == "__main__":
    with Pool() as pool:
        results = pool.map(run_test, generate_params(param_options))
    results.sort(key=lambda x: x[0])
    save_results_to_file(results)
    print("best result: ",results[-1])
