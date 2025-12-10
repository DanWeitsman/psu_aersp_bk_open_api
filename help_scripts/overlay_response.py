#!/usr/bin/env python3

import os
from help_functions import *
import argparse
import matplotlib.pyplot as plt
from scipy.signal import welch,csd
from scipy.fft import fft,ifft

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ["Times New Roman"]
plt.rcParams['font.size'] = 12

#%%%

cases = ['5_12/r1','5_14/r14']

df = 1
overlap = 0.5
window = 'hann'
s = 0.0366167
l = .1380267
sos = 343.56


data = {}
for case in cases:

    dat = read_data_h5(os.path.join(os.getcwd(),case,os.path.basename(case)+'.h5'))
    acs_data = np.asarray([dat[key]['scaled_samples'] for key in dat.keys()])

    fs = dat['channel0']['sample_rate']
    nperseg=  fs/df
    f,Gxx = welch(acs_data, fs=fs, window=window, nperseg=nperseg, noverlap=int(overlap*nperseg), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
    f,Gxx_educed = psd_at_sample(acs_data[0],acs_data[1],fs,df,s = s,l = l,sos = sos,overlap =overlap,window = window)
    R,Z,alpha = acs_response(acs_data[0],acs_data[1],fs = fs,df = df,s = s,l =l,sos = sos,overlap = overlap,window = window)
    data.update({os.path.basename(case):{'acs_data':acs_data,'f':f,'Gxx':Gxx,'Gxx_educed':Gxx_educed,'R':R,'Z':Z,'alpha':alpha}})

#%%

fig,ax = plt.subplots(3,1, figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.125,hspace = 0.3)
for k,v in data.items():
    for i in range(3):
        ax[i].plot(f[1:],10*np.log10(v['Gxx'][i,1:]/20e-6**2))
    ax[-1].plot(f[1:],10*np.log10(v['Gxx_educed'][1:]/20e-6**2))

for i in range(3):
    ax[i].grid()
    ax[i].axis([0,4e3,25,75])
    ax[i].set_title(f'Channel {i+1}')
    if i !=2:
        ax[i].set_xticklabels([])

ax[1].set_ylabel('$PSD, \ dB/Hz \ (re: \ 20 \mu Pa)$')
ax[-1].set_xlabel('$Frequency \ [Hz]$')

fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))

for k,v in data.items():
    ax[0].plot(f,np.real(v['Z']))
    ax[1].plot(f,np.imag(v['Z']))

ax[0].tick_params(axis = 'x', labelsize=0)
ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
ax[0].set_xlim([0,4e3])
ax[0].set_ylim([-100,100])
ax[0].grid()

ax[1].set_ylabel(r'$Reactance, \ \overline{\chi}$')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_xlim([0,4e3])
ax[1].set_ylim([-100,100])
ax[1].grid()

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
for k,v in data.items():
    ax.plot(f,np.abs(v['R']))
ax.set_ylabel(r'$Absorption, \ \alpha$')
ax.set_xlim([0,4e3])
ax.set_ylim([0,1])
ax.grid()
# ax.legend(['Original','Corrected'])
