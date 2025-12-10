#!/usr/bin/env python3
# pylint: disable=C0103

import os
from shutil import rmtree
import argparse
from streaming_single_module import streaming_single_module
from streaming_interpretation import streaming_interpretation
from plot import plot

#%%

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("addr", help="IP address of the LAN-XI module")
    parser.add_argument("-n", "--name", default="measurement", \
        help="Name of the measurement")
    parser.add_argument("-d", "--save_dir", default="./", \
        help="Relative path from current directory where to save the data")
    parser.add_argument("-fs", "--sampling_rate", default=20e3, type=float, \
        help="The sampling rate (in Hz) of the measurement")
    parser.add_argument("-t", "--time", default=10, type=int, \
        help="The time (in seconds) of the measurement")
    parser.add_argument("-c", "--calibrate",action="store_true", \
        help="Provide this argument to output raw voltage instead of pressure when calibrating the microphones.")
    parser.add_argument("-s", "--sensitivity",type=float, nargs='+', \
        help="Manually provided microphone sensitivities [mV/Pa] are only necessary when TEDS information is unavaliable or if the actual sensitivities differ from the TEDS data. Provide as list with a length corresponding to the number of channels being used.")
    args = parser.parse_args()
    parser.add_argument("-p", "--plot", action="store_true", \
        help="Include to plot time series and spectrum")
    args = parser.parse_args()
    args = vars(args)

    args['save_dir'] = os.path.join(os.getcwd(),args['save_dir'],args['name'])
    args['file']=args['name']+'.stream'

    if os.path.exists(args['save_dir']):
        rmtree(args['save_dir'])
    os.mkdir(args['save_dir'])

    streaming_single_module(**args)
    streaming_interpretation(**args)

    if args['plot']:
        plot(**args)

if __name__ == "__main__":
    main()