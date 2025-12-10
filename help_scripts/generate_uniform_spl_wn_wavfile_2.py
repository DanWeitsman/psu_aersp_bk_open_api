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
parser.add_argument("-df", default=1, type=float, \
    help="Frequency resolution (Hz)")
parser.add_argument("-fs", "--fs_src", default=44.1e3, type=float, \
    help="computer audio sampling rate [Hz]")
parser.add_argument("-A", default=0.1, type=float, \
    help="amplitude of generated signal, specified as a percentage of the system's output level")
parser.add_argument("-fc",nargs='+', default=(100,4500), type=float, \
    help="lower and upper cutoff frequencies of the bandpass filter applied to the test signal [Hz]")
parser.add_argument("-s", default=0.0366167, type=float, \
    help="microphone spacing [m]")
parser.add_argument("-l", default=.1380267, type=float, \
    help="distance between the sample and the mic that is closer to the driver [m]")
parser.add_argument("-sos", default=343.563, type=float, \
    help="speed of sound [m/s]")
parser.add_argument("-ovr", "--overlap", default=0, type=float, \
    help="percentage overlap between records")
parser.add_argument("-w", "--window", dest="window", default="boxcar", \
    help="window function")
args = parser.parse_args()

#%%

data = read_data_h5(os.path.join(os.getcwd(),args.save_dir,args.name,f'{args.name}.h5'))
keys = list(data.keys())
acs_data = np.asarray([data[key]['scaled_samples'] for key in data.keys()])
# sampling rate of acquired acoustic data [Hz]
fs1 = data['channel0']['sample_rate']

window = 'boxcar'

nperseg = len(acs_data[0])
args.df = fs1/nperseg

f,Gxx = psd_at_sample(acs_data[0],acs_data[1],fs = fs1,df = args.df,s = args.s,l = args.l,sos = args.sos,overlap = args.overlap,window = args.window)
Gxx[np.isnan(Gxx)] = 0

uniform_spl_filt_resp = np.mean(Gxx)/Gxx*fs1/args.fs_src
uniform_spl_filt_resp[np.isnan(uniform_spl_filt_resp)] = 0

xn = psd_noise(psd = uniform_spl_filt_resp,A = args.A,fs = fs1,fc =args.fc)
xn_upsampled = upsample(xn,fs1 = fs1,fs2 = args.fs_src)
wavfile.write(os.path.join(os.path.dirname(__file__),"WN_filt_2.wav"), int(args.fs_src), xn_upsampled)

t1 = np.arange(len(xn))*fs1**-1
t2 = np.arange(len(xn_upsampled))*args.fs_src**-1


fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
plt.subplots_adjust(left = .15,bottom = .15)
ax.plot(t1,xn)
ax.plot(t2,xn_upsampled,linestyle = '-.')
ax.set_ylabel('$Pressure [Pa]$')
ax.set_xlabel('Time [s]')
ax.grid()

leglab = []
fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.125)
for i in range(len(acs_data)):
    ax.plot(f[1:],10*np.log10(G_xx_measured[i][1:]/20e-6**2))
    leglab.append(f'Measured: Channel {i+1}')
ax.plot(f[1:],10*np.log10(Gxx[1:]/20e-6**2),linestyle = ':')
ax.legend(leglab+['Educed'])
ax.set_ylabel('$PSD, \ dB/Hz \ (re: \ 20 \mu Pa)$')
ax.set_xlabel('$Frequency \ [Hz]$')
# ax.set_ylim([0,100])
ax.grid()

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.125)
ax.plot(f[1:],Gxx[1:])
ax.plot(f[1:],Gxx[1:]*np.mean(Gxx[1:])/Gxx[1:],linestyle = ':')
ax.set_ylabel('$PSD, \ dB/Hz \ (re: \ 20 \mu Pa)$')
ax.set_xlabel('$Frequency \ [Hz]$')
# ax.set_ylim([0,np.mean(Gxx[1:])*10])
ax.grid()


