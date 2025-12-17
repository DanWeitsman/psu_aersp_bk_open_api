#!/usr/bin/env python3

import numpy as np
import os
from scipy.signal import welch,sosfilt,butter
from help_functions import *
import json
import matplotlib.pyplot as plt
from scipy.fft import fft
import argparse

#%%

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--correction", default=0,type=float, \
    help="Barometric pressure correction factor to add to pistonephone calibration [dB]")
parser.add_argument("-pist", "--pistonphone", action="store_true", \
        help="Include when using a pistonphone calibrator")
parser.add_argument("-p", "--plot", action="store_true", \
        help="Include to plot time series and spectrum")
args = parser.parse_args()

def compute_sens(data,df):
    k = list(data.keys())[0]
    dt = data[k]['sample_rate']**-1
    nperseg = (df*dt)**-1

    f, G = welch(data[k]['scaled_samples'], fs=data[k]['sample_rate'], window='hann', nperseg=nperseg, noverlap=int(noverlap*nperseg), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean') 
    if args.pistonphone:
        S = np.sqrt(np.trapezoid(G[int((cal_freq-df*4)/df):int((cal_freq+df*10)/df)], dx = df))/(10**((124.03+args.correction)/20)*20e-6)
    else:
        S = np.sqrt(np.trapezoid(G[int((cal_freq-df*4)/df):int((cal_freq+df*10)/df)], dx = df))


    # b,a = butter(2, Wn =[cal_freq-df*5,cal_freq+df*5], btype='bandpass', analog=False, output='ba',fs = data[k]['sample_rate'])
    # filt_data = lfilter(b,a, data[k]['scaled_samples'])
    # S = np.sqrt(np.mean(filt_data**2))/(10**((124.03+args.correction)/20)*20e-6)
    
    return S
#%%

files_in_dir = np.asarray(os.listdir(os.path.join(os.getcwd())))
f_name =files_in_dir[[os.path.isdir(path) for path in files_in_dir]]

data = {}
[data.update({f:read_data_h5(os.path.join(os.getcwd(),f,f+'.h5'))}) for f in f_name]

df = 5
noverlap = 0.5

if args.pistonphone:
    cal_freq = 250
else:
    cal_freq = 1e3

sensitivity = {}
[sensitivity.update({k:compute_sens(v,df)}) for k,v in data.items()]
[print(f"{k}: {v}") for k,v in sensitivity.items()]

if args.plot:

    for file in f_name:
        
        dat = data[file][list(data[file].keys())[0]]
        dt = dat['sample_rate']**-1
        t = np.arange(len(dat['scaled_samples']))*dt
        nperseg = (df*dt)**-1

        # print(10*np.log10(np.mean((dat['scaled_samples']/sensitivity[file])**2)/20e-6**2))

        f, G = welch(dat['scaled_samples']/sensitivity[file], fs=dat['sample_rate'], window='hann', nperseg=nperseg, noverlap=int(noverlap*nperseg), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')

        fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
        plt.subplots_adjust(left = .15,bottom = .15)
        ax[0].plot(t,dat['scaled_samples'])
        ax[0].set_ylabel('$Pressure [Pa]$')
        ax[0].set_xlabel('Time [s]')
        ax[0].grid()
        ax[1].plot(f,10*np.log10(G*df/20e-6**2))
        ax[1].set_ylabel('SPL [Pa]')
        ax[1].set_xlabel('Frequency [Hz]')
        ax[1].grid()
        plt.savefig(f'{file}_cal.png',format = 'png')
        plt.close()



with open(os.path.join(os.path.dirname(__file__),'mic_sens.json'),'w') as f:
    json.dump(sensitivity,f,indent=2)
