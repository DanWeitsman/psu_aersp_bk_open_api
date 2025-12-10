#!/usr/bin/env python3

import numpy as np
from help_functions import *
import os
import argparse
from scipy.io import wavfile 
import matplotlib.pyplot as plt

#%%

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", dest="name", default="measurement", \
    help="Name of the measurement")
parser.add_argument("-d", "--save_dir", dest="save_dir", default="./", \
    help="Absolute path of where to save the data")
parser.add_argument("-fs", "--fs_src", default=44.1e3, type=float, \
    help="computer audio sampling rate [Hz]")
parser.add_argument("-A", default=0.1, type=float, \
    help="amplitude of generated signal, specified as a percentage of the system's output level")
parser.add_argument("-fc",nargs='+', default=(300,4000), type=float, \
    help="lower and upper cutoff frequencies of the bandpass filter applied to the test signal [Hz]")
parser.add_argument("-s", default=0.0366167, type=float, \
    help="microphone spacing [m]")
parser.add_argument("-l", default=.1380267, type=float, \
    help="distance between the sample and the mic that is closer to the driver [m]")
parser.add_argument("-sos", default=343.563, type=float, \
    help="speed of sound [m/s]")
parser.add_argument("-p", "--plot", action="store_true", \
    help="Include to plot time series and spectrum")
args = parser.parse_args()

#%%

data = read_data_h5(os.path.join(os.getcwd(),args.save_dir,args.name,f'{args.name}.h5'))
keys = list(data.keys())
acs_data = np.asarray([data[key]['scaled_samples'] for key in data.keys()])
# sampling rate of acquired acoustic data [Hz]
fs1 = data['channel0']['sample_rate']

uniform_spl_filt_resp = uniform_spl_filt_response(m1 =acs_data[0] ,m2 =acs_data[1],fs = fs1,fs_src = args.fs_src,s = args.s,l = args.l,sos = args.sos)
xn = psd_noise(psd = uniform_spl_filt_resp,A = args.A,fs = args.fs_src,fc =args.fc)

# xn_wn = white_noise(T = 20,A = args.A,fs = args.fs_src,fc =args.fc)

if args.plot:

    f,Gxx = welch(xn, fs=args.fs_src, window='boxcar', nperseg=len(xn), noverlap=int(0), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')

    leglab = []
    fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
    plt.subplots_adjust(bottom = 0.125)
    ax.plot(uniform_spl_filt_resp)
    ax.set_ylabel('$PSD, \ dB/Hz \ (re: \ 20 \mu Pa)$')
    ax.set_xlabel('$Frequency \ [Hz]$')
    ax.set_xlim([0,20e3])
    # ax.set_ylim([0,args.A])
    ax.grid()
    plt.savefig(os.path.join(os.path.dirname(__file__),f'input_signal.png'),format = 'png')


wavfile.write(os.path.join(os.path.dirname(__file__),"WN_filt.wav"), int(args.fs_src), xn)







