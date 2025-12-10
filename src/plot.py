#!/usr/bin/env python3
# pylint: disable=C0103

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy.signal import welch

#%%

def plot(**kwargs):
    data = {}
    with h5py.File(kwargs['file'], "r") as f:
        for k, v in f.items():
            if len(v)>1:
                data_temp = {}
                for k_2, v_2 in v.items():
                    data_temp = {**data_temp, **{k_2: v_2[()]}}
                data = {**data,**{k:data_temp}}
            else:
                data = {**data, **{k: v[()]}}

    #%%
    fig, ax = plt.subplots(len(data), 1)

    for indx, (k, v) in enumerate(data.items()):

        df = 10
        dt = v['sample_rate']**-1
        N = (df*dt)**-1
        fs = v['sample_rate']
        f,pxx = welch(v['scaled_samples'], fs=fs, window='hann', nperseg=N, noverlap=int(N/2), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
        spl = 10*np.log10(pxx*df/20e-6**2)

        ax[indx].plot(f, spl)
        ax[indx].set_title(f'{k}')
        ax[indx].set_ylabel(f'$SPL \ dB, \ (re: \ 20 \mu Pa)$')
        ax[indx].set_xlabel(f'$Frequency \ [Hz]$')
        ax[indx].grid()

    plt.savefig(os.path.join(os.path.dirname(kwargs['file']),'spectrum.png'),format = 'png')


    #%%

    fig, ax = plt.subplots(len(data), 1)

    for indx, (k, v) in enumerate(data.items()):

        t = np.arange(len(v['scaled_samples']))*v['sample_rate']**-1

        ax[indx].plot(t, v['scaled_samples'])
        ax[indx].set_title(f'{k}')
        ax[indx].set_ylabel(f'Pressure [Pa]')
        ax[indx].set_xlabel(f'Time [s]')
        ax[indx].grid()
        # ax[indx].set_xlim([0,1000*dt])

    plt.savefig(os.path.join(os.path.dirname(kwargs['file']),'p_tseries.png'),format = 'png')


