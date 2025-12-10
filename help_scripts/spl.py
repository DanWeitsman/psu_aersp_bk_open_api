#!/usr/bin/env python3

import os
from help_functions import *
import argparse

#%%

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", dest="name", default="measurement", \
    help="Name of the measurement")
parser.add_argument("-d", "--save_dir", dest="save_dir", default="./", \
    help="Absolute path of where to save the data")
args = parser.parse_args()

data = read_data_h5(os.path.join(os.getcwd(),args.save_dir,args.name,f'{args.name}.h5'))
keys = list(data.keys())

def calculate_spl(x):
    return np.round(20*np.log10(np.sqrt(np.mean(x**2))/20e-6),1)

spl = list(map(lambda k: calculate_spl(data[k]['scaled_samples']),keys))

for i,spl_iter in enumerate(spl):
    print(f'Channel {i+1}: {spl_iter}')