#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from help_functions import *
import os
import argparse

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ["Times New Roman"]
plt.rcParams['font.size'] = 14
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyle = ['-',':','--','-.']


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", dest="name", default="measurement", \
        help="Name of the measurement")
    parser.add_argument("-d", "--save_dir", dest="save_dir", default="./", \
        help="Absolute path of where to save the data",default = 5)
    parser.add_argument("-df", type=float, \
        help="Frequency resolution (Hz)")
    parser.add_argument("-s", default=0.0366167, type=float, \
        help="microphone spacing [m]")
    parser.add_argument("-l", default=.1380267, type=float, \
        help="distance between the sample and the mic that is closer to the driver [m]")
    parser.add_argument("-sos", default=343.563, type=float, \
        help="speed of sound [m/s]")
    parser.add_argument("-ovr", "--overlap", default=0.5, type=float, \
        help="percentage overlap between records")
    parser.add_argument("-w", "--win", default="hann", \
        help="window function")
    args = parser.parse_args()

    data = read_data_h5(os.path.join(os.getcwd(),args.save_dir,args.name,f'{args.name}.h5'))
    max_ind = np.asarray([len(data[key]['scaled_samples']) for key in data.keys()]).min()
    acs_data = np.asarray([data[key]['scaled_samples'][:max_ind] for key in data.keys()])

    # sampling rate [Hz]
    fs = data['channel0']['sample_rate']
    # temporal resolution [s]
    dt = fs**-1

    # if frequency resolution is not provided default corresponds to narrowband resolution
    if args.df is None:
        args.df = (acs_data.shape[-1]*dt)**-1

    R,Z,alpha = acs_response(m1 =acs_data[0],m2 = acs_data[1],fs = fs,df = args.df,s = args.s,l = args.l,sos = args.sos,overlap = args.overlap,window = args.win)
    f = np.arange(len(R))*args.df

#%%
    fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
    plt.subplots_adjust(left = 0.15,bottom=0.15,hspace = 0.3)
    ax[0].plot(f,np.real(Z))
    ax[1].plot(f,np.imag(Z))
    ax[0].set(xticklabels = [],ylabel= r'$Resistance, \ \overline{\theta}$',xlim =[0,4e3],ylim = [-10,10])
    ax[1].set(ylabel= r'$Reactance, \ \overline{\chi}$',xlabel = r'$Frequency \ [Hz]$',xlim =[0,4e3],ylim = [-10,10])
    ax[0].grid()
    ax[1].grid()
    plt.savefig(os.path.join(os.getcwd(),args.save_dir,args.name,f'Z.png'),format = 'png')

    fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
    plt.subplots_adjust(left = 0.15,bottom=0.15,hspace = 0.3)
    ax[0].plot(f,np.abs(R))
    ax[1].plot(f,alpha)
    ax[0].set(xticklabels = [],ylabel= r'$Reflection, \ |\mathit{R}|$',xlim =[100,4e3],ylim = [0,1])
    ax[1].set(ylabel= r'$Absorption, \ \alpha$',xlabel = r'$Frequency \ [Hz]$',xlim =[100,4e3],ylim = [0,1])
    ax[0].grid()
    ax[1].grid()
    plt.savefig(os.path.join(os.getcwd(),args.save_dir,args.name,f'alpha.png'),format = 'png')

if __name__ == "__main__":
    main()