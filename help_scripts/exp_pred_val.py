#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from help_functions import *
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__,'../../dependencies/resonator')))
import resonator as res

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ["Times New Roman"]
plt.rcParams['font.size'] = 12

#%%

exp_data_path = '//Users/danielweitsman/Documents/research/NIT_testing/3_25/oar_bends/data/3_5_25/'
exp_case_name = 'r1'

# microphone spacing [m]
s = 0.0366167
# distance to the sample [m]
l = .1380267

# sos [m/s]
sos = 20.047*np.sqrt(273.15+20.556)
# density [kg/m^3]
rho = 1.125

# frequency resolution [Hz]
df = 2
# percentage overlap between records
noverlap = 0.5

# Inner cross-sectional area of the NIT [m^2].
A = 0.050635**2
# A_ratio = 1

# Number of resonators
N = 22**2
# length of resonator [m]
L = 0.085
# radius of resonator [m]
r = 0.00085
# Open area ratio of the sample
OAR = N*np.pi*r**2/A

#%%

data = read_data_h5(os.path.join(exp_data_path,exp_case_name,f'{exp_case_name}.h5'))
# sampling rate [Hz]
fs = data['channel0']['sample_rate']
# temporal resolution [s]
dt = fs**-1

# samples per record
nperseg = (df*dt)**-1
H =  TF(data['channel0']['scaled_samples'],data['channel1']['scaled_samples'],fs = fs,nperseg=nperseg,noverlap = noverlap)

f = np.arange(len(H))*df
# wave number 
k = 2*np.pi*f/sos

# acoustic transfer function of the incident acoustic wave between the two micriophones
H_i = np.exp(-1j*k*s)
# acoustic transfer function of the reflected acoustic wave between the two micriophones
H_r = np.exp(1j*k*s)

# Reflection coefficient at the first mic (closer to the driver)
R_1 = (H-H_i)/(H_r-H)
# Reflection coefficient of the test sample
R_exp = R_1*np.exp(1j*2*k*l)
# Acoustic impedance of the test sample
Z_exp = (1+R_exp)/(1-R_exp)
# absorption coefficient of the test sample
alpha_exp =  1 - abs(R_exp)**2
# alpha = 4* np.real(Z)/((np.real(Z)+1)**2+np.imag(Z)**2)

#%%

res_pred = res.resonator(a_n = r,L_n = L/2,a_c = r,L_c =  L/2,c = sos,rho = rho)
res_pred.set_Z(f[1:])
Z_pred = res_pred.Z/OAR
R_pred = (Z_pred-1)/(Z_pred+1)
alpha_pred =  1 - abs(R_exp)**2

#%%

fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.15,left = 0.15)
ax[0].set_xticklabels([])
ax[0].plot(f[1:],np.abs(R_pred))
ax[0].plot(f,np.abs(R_exp))
ax[0].set_ylabel(r'$|\mathcal{R}|$')
ax[0].set_xlim([0,4e3])
ax[0].set_ylim([0,1])
ax[0].legend(['Predicted','Measured'])
ax[0].grid()

ax[1].plot(f[1:],np.unwrap(np.angle(R_pred)))
ax[1].plot(f,np.unwrap(np.angle(R_exp)))
ax[1].set_ylabel(r'$\phi \ [rad]$')
ax[-1].set_xlabel('Frequency [Hz]')
ax[-1].grid()
ax[-1].set_xlim([0,4e3])
ax[-1].set_ylim([-np.pi*5,np.pi/2])


fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.15,left = 0.15)
ax[0].set_xticklabels([])
ax[0].plot(f[1:],np.real(Z_pred))
ax[0].plot(f,np.real(Z_exp))
ax[0].set_ylabel(r'$\overline{\theta}$')
ax[0].set_xlim([0,4e3])
ax[0].set_ylim([0,10])
ax[0].legend(['Predicted','Measured'])
ax[0].grid()

ax[1].plot(f[1:],np.imag(Z_pred))
ax[1].plot(f,np.imag(Z_exp))
ax[1].set_ylabel(r'$\overline{\chi}$')
ax[-1].set_xlabel('Frequency [Hz]')
ax[-1].grid()
ax[-1].set_xlim([0,4e3])
ax[-1].set_ylim([-5, 5])
