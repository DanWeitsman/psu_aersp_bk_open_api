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

def calculate_spl(x):
    return np.round(10*np.log10(np.mean(x**2)/20e-6**2),1)


P = np.asarray([0.0002199,0.000924,0.00182,0.004551,0.0129948,0.0268226,0.0366375,0.064614,0.10744,0.17407,0.243,0.3276,0.4428,0.5621,0.648,0.73272,0.7884,0.69138,2.0016,5.4,9.4146])

files_in_dir = np.asarray(os.listdir(os.path.join(os.getcwd())))
f_name =files_in_dir[[os.path.isdir(path) for path in files_in_dir]]
f_name = sorted(list(f_name), key=lambda x: int(x[1:]))

data = {}
[data.update({f:read_data_h5(os.path.join(os.getcwd(),f,f+'.h5'))}) for f in f_name]

#%%

# microphone spacing [m]
s = 0.0366167
# distance to the sample [m]
l = .1380267

# sos [m/s]
sos = 20.047*np.sqrt(273.15+20.556)
# density [kg/m^3]
rho = 1.125
# percentage overlap between frequency bins
noverlap = 0
# windowing function
window = 'boxcar'

SPL_measured = np.zeros((len(f_name),3))
SPL_educed = np.zeros(len(f_name))

for i,run in enumerate(f_name):
    
    acs_data = np.asarray([data[run]['channel0']['scaled_samples'],data[run]['channel1']['scaled_samples'],data[run]['channel2']['scaled_samples']])

    nperseg = acs_data.shape[-1]
    # sampling rate [Hz]
    fs = data[run]['channel0']['sample_rate']
    # temporal resolution [s]
    dt = fs**-1
    # frequency resolution [Hz]
    df = (dt*nperseg)**-1
    # frequency array [Hz]
    f = np.arange(int(nperseg/2)+1)*df
    # wave number [m^-1]
    k = 2*np.pi*f/sos

    # acoustic transfer function of the incident acoustic wave between the two microphones (assuming plane wave propagation without thermoviscous lossess)
    H_i = np.exp(-1j*k*s)
    # acoustic transfer function of the reflected acoustic wave between the two microphones
    H_r = np.exp(1j*k*s)

    Gxy = csd(acs_data[0], acs_data[1], fs=fs, window=window, nperseg=nperseg, noverlap=int(noverlap*nperseg), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')[-1]
    Gxx = welch(acs_data, fs=fs, window=window, nperseg=nperseg, noverlap=int(noverlap*nperseg), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')[-1]

    # transfer function between the two microphone signals 
    H_12 = Gxy/Gxx[0]
    # Reflection coefficient at the first mic (closer to the driver)
    R_1 = (H_12-H_i)/(H_r-H_12)
    # transfer function between the first mic (closest to the driver) and the test sample
    H_1s = ((R_1*np.exp(1j*k*l)+np.exp(-1j*k*l))/(1+R_1))

    # build filter response
    filt_resp = np.ones(nperseg,dtype = complex)
    filt_resp[:int(nperseg/2)+1] = H_1s
    if nperseg%2:
        filt_resp[int(nperseg/2)+1:] = np.conj(H_1s[1:])[::-1]
    else:
        filt_resp[int(nperseg/2)+1:] = np.conj(H_1s[1:-1])[::-1]

    Xm_filt = fft(acs_data[0])*dt*filt_resp
    Sxx_filt = (nperseg*dt)**-1*np.abs(Xm_filt)**2
    Gxx_filt = Sxx_filt[...,:int(nperseg/2)+1]
    if nperseg%2:
        Gxx_filt[...,1:] = 2*Gxx_filt[...,1:]
    else:
        Gxx_filt[...,1:-1] = 2*Gxx_filt[...,1:-1]
    
    SPL_measured[i] = 10*np.log10(np.trapezoid(Gxx[:,1:], dx = df,axis = -1)/20e-6**2).T
    SPL_educed[i] = 10*np.log10(np.trapezoid(Gxx_filt[1:], dx = df)/20e-6**2)
    print(f'dSPL = {np.round(SPL_measured[i,-1]-SPL_educed[i],2)}')

#%%
fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
plt.subplots_adjust(left = .125,bottom = .125)
ax.plot(P,SPL_measured,marker = 'o')
ax.plot(P,SPL_educed,marker = '^',c = 'black',linewidth = 0)
ax.set_ylabel("$ SPL,\ dB\ (re:20 \mu Pa) $")
ax.set_xlabel("$ Input \ Power \ [W]$")
ax.legend(['Mic 1 (Measured)','Mic 2 (Measured)','Mic 3 (Measured)','Educed'])
ax.set_xlim([0.0001,10])
ax.grid()
ax.set_xscale('log')

