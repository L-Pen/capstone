import numpy as np
def save_results_to_file(results):
    with open("results.txt","w") as f:
        for result in results:
            f.write("%s\n" % str(result))
    
    print("results saved to results.txt")


def generate_params(params):
    params_list = []
    for key in params:
        if isinstance(params[key],list):
            if len(params_list) == 0:
                for value in params[key]:
                    params_list.append({key:value})
            else:
                new_params_list = []
                for param in params_list:
                    for value in params[key]:
                        new_param = param.copy()
                        new_param[key] = value
                        new_params_list.append(new_param)
                params_list = new_params_list
        else:
            if len(params_list) == 0:
                params_list.append({key:params[key]})
            else:
                for param in params_list:
                    param[key] = params[key]
    return params_list