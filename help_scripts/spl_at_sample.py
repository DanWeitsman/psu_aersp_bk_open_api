#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from help_functions import *
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", dest="name", default="measurement", \
    help="Name of the measurement")
parser.add_argument("-d", "--save_dir", dest="save_dir", default="./", \
    help="Absolute path of where to save the data")
parser.add_argument("-df", default=1, type=float, \
    help="Frequency resolution (Hz)")
parser.add_argument("-s", default=0.0366167, type=float, \
    help="microphone spacing [m]")
parser.add_argument("-l", default=.1380267, type=float, \
    help="distance between the sample and the mic that is closer to the driver [m]")
parser.add_argument("-sos", default=343.563, type=float, \
    help="speed of sound [m/s]")
parser.add_argument("-ovr", "--overlap", default=0, type=float, \
    help="percentage overlap between records")
parser.add_argument("-w", "--win", dest="window", default="boxcar", \
    help="window function")
parser.add_argument("-m", "--m3",action="store_true", \
    help="Provide this argument if there is a microphone flush mounted to the test sample for comparison.")
parser.add_argument("-p", "--plot", action="store_true", \
    help="Include to plot time series and spectrum")
args = parser.parse_args()

data = read_data_h5(os.path.join(os.getcwd(),args.save_dir,args.name,f'{args.name}.h5'))
keys = list(data.keys())
acs_data = np.asarray([data[key]['scaled_samples'] for key in data.keys()])

# sampling rate [Hz]
fs = data['channel0']['sample_rate']
# temporal resolution [s]
dt = fs**-1

# samples per record
nperseg = int(np.round((args.df*dt)**-1))
# number of points to overlap 
noverlap = int(np.round(args.overlap*nperseg))

f,Gxx_educed =  psd_at_sample(acs_data[0],acs_data[1],fs = fs,df = args.df,s = args.s,l = args.l,sos = args.sos,overlap = 0.0,window = args.window)
SPL_educed = 10*np.log10(np.trapezoid(Gxx_educed[1:], dx = args.df)/20e-6**2)
_,G_xx_measured = welch(acs_data, fs=fs, window=args.window, nperseg=nperseg, noverlap=int(noverlap*nperseg), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')

if args.m3:
    SPL_measured = 10*np.log10(np.trapezoid(G_xx_measured[-1][1:], dx = args.df)/20e-6**2)
    print(f'dSPL = {np.round(SPL_measured-SPL_educed,2)}')
else:
    print(f"SPL at sample = {np.round(SPL_educed,2)}")

if args.plot:
    leglab = []
    fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
    plt.subplots_adjust(bottom = 0.125)
    for i in range(len(G_xx_measured)):
        ax.plot(f[1:],10*np.log10(G_xx_measured[i][1:]/20e-6**2))
        leglab.append(f'Measured: Channel {i+1}')
    ax.plot(f[1:],10*np.log10(Gxx_educed[1:]/20e-6**2),linestyle = ':')
    ax.legend(leglab+['Educed'])
    ax.set_ylabel('$PSD, \ dB/Hz \ (re: \ 20 \mu Pa)$')
    ax.set_xlabel('$Frequency \ [Hz]$')
    # ax.set_ylim([0,100])
    ax.grid()
    plt.savefig(os.path.join(os.getcwd(),args.save_dir,args.name,f'spl_at_sample.png'),format = 'png')



