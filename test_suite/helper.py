import numpy as np
import pandas as pd
from keras.utils import to_categorical 
def avg(group, group_size): 
    group['GroupNumber'] = np.array(range(len(group.index))) // group_size
    res = group.groupby('GroupNumber').mean()
    return res
def load_data(group_size):
    data = pd.read_hdf("../../initialSingleCellDf-channel-20220916-MW_018-001.h5", key="df")
    #print all the columns
    ANTIGENS = ['null', 'E1', 'G4', 'V4', 'T4', 'Q4', 'A2', 'N4']
    data = data.loc[(data.index.get_level_values('CellType') == 'OT-1') & 
                       (data.index.get_level_values('Peptide').isin(ANTIGENS))
                    ]
    data = data.groupby(['Peptide', 'Time', 'Replicate', 'Concentration']).apply(avg, group_size=group_size)
    antigen = list(data.index.get_level_values('Peptide'))
    X = np.array(data.values)
    y = np.array(list(map(lambda x: ANTIGENS.index(x), antigen)))
    y = to_categorical(y)
    return X,y

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