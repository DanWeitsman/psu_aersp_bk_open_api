import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy.signal import welch,csd
from help_functions import *

#%%

# absolute path to h5 file containing acoustic data
data_fpath = os.path.join(os.path.dirname(__file__),'..','dan_impedance','std_cal_100dB_1.h5')

# absolute path to h5 file containing acoustic data
std_cal_fpath = os.path.join(os.path.dirname(__file__),'..','dan_impedance','std_cal_100dB_3.h5')
# absolute path to h5 file containing acoustic data
switch_cal_fpath = os.path.join(os.path.dirname(__file__),'..','dan_impedance','switch_cal_100dB_3.h5')

# microphone spacing [m]
s = 0.0366167
# distance to the sample [m]
l = .10141

t_lim = [5,10]
# frequency resolution [Hz]
df = 5
# sos [m/s]
c = 343
# density [kg/m^3]
rho = 1.125

#%%

data,std_cal_data,switch_cal_data = list(map(lambda f: read_data_h5(f),[data_fpath,std_cal_fpath,switch_cal_fpath]))

# temporal resolution [s]
dt = data['Table1']['Ds1-Time'][1]-data['Table1']['Ds1-Time'][0]
# sampling rate [Hz]
fs = dt**-1
# samples per record
nperseg = (df*dt)**-1
# percentage overlap between records
noverlap = 0.5

t_lim_ind = slice(int(t_lim[0]*fs),int(t_lim[-1]*fs))

# H_12,H_std,H_switch =  list(map(lambda dat: TF(dat['Table1']['Ds2-Signal 1'][t_lim_ind],dat['Table1']['Ds3-Signal 2'][t_lim_ind],fs = fs,nperseg=nperseg,noverlap = noverlap),[data,std_cal_data,switch_cal_data]))

H_12 = TF(data['Table1']['Ds2-Signal 1'][t_lim_ind],data['Table1']['Ds3-Signal 2'][t_lim_ind],fs = fs,nperseg=nperseg,noverlap = noverlap)
H_std= TF(std_cal_data['Table1']['Ds2-Signal 1'][t_lim_ind],std_cal_data['Table1']['Ds3-Signal 2'][t_lim_ind],fs = fs,nperseg=nperseg,noverlap = noverlap)
H_switch= TF(switch_cal_data['Table1']['Ds2-Signal 1'][t_lim_ind],switch_cal_data['Table1']['Ds3-Signal 2'][t_lim_ind],fs = fs,nperseg=nperseg,noverlap = noverlap)

#H_c_mag = (abs(H_std)*abs(H_switch))**.5
# H_c_ph = .5*(np.angle(H_switch)+np.angle(H_std))
# H_c_2 = H_c_mag*np.exp(1j*H_c_ph)
# f, H2 = welch(data['Table1']['Ds2-Signal 1'][t_lim_ind], fs=fs, window='hann', nperseg=nperseg, noverlap=int(noverlap*nperseg), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')

H_c = (H_std*H_switch)**0.5
H =H_12


#%%

# frequency array [Hz]
f = np.arange(len(H))*df
# wave number [m^-1]
k = 2*np.pi*f/c
# acoustic transfer function of the incident acoustic wave between the two micriophones
H_i = np.exp(-1j*k*s)
# acoustic transfer function of the reflected acoustic wave between the two micriophones
H_r = np.exp(1j*k*s)

# Reflection coefficient at the first mic (closer to the driver)
R_1 = (H-H_i)/(H_r-H)
# Reflection coefficient of the test sample
R = R_1*np.exp(1j*2*k*(l+s))


R_mag = ((1+abs(H)**2-2*abs(H)*np.cos(np.angle(H)+k*s))/(1+abs(H)**2-2*abs(H)*np.cos(np.angle(H)-k*s)))**0.5
phi_2 = 2*k*(l+s)+np.arctan((2*abs(H)*np.cos(np.angle(H))*np.sin(k*s)-np.sin(2*k*s))/(abs(H)**2-2*abs(H)*np.cos(np.angle(H))*np.cos(k*s)+np.cos(2*k*s)))
R2 = -R_mag*np.exp(1j*phi_2)

D = 1+np.real(H)**2+np.imag(H)**2-2*(np.real(H)*np.cos(k*s)+np.imag(H)*np.sin(k*s))
R_r = (2*np.real(H)*np.cos(k*(2*l+s))-np.cos(2*k*l)-(np.real(H)**2+np.imag(H)**2)*np.cos(2*k*(l+s)))/D
R_i = (2*np.real(H)*np.sin(k*(2*l+s))-np.sin(2*k*l)-(np.real(H)**2+np.imag(H)**2)*np.sin(2*k*(l+s)))/D
R3 = R_r+1j*R_i

# Acoustic impedance of the test sample
Z = (1+R)/(1-R)
# absorption coefficient of the test sample
alpha =  1 - abs(R)**2
resist = alpha/(2*(1-np.real(R))-alpha)
react = 2*alpha*np.imag(R)/(2*(1-np.real(R))-alpha)

# alpha = 4* np.real(Z)/((np.real(Z)+1)**2+np.imag(Z)**2)

#%%

fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
ax[0].tick_params(axis = 'x', labelsize=0)
ax[0].plot(f,np.real(Z))
ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
ax[0].set_xlim([500,5e3])
# ax[0].set_ylim([-1,1])
ax[0].grid()

ax[1].plot(f,np.imag(Z))
ax[1].set_ylabel(r'$Reactance, \ \overline{\chi}$')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_xlim([500,5e3])
# ax[1].set_ylim([-10,10])
ax[1].grid()
# plt.savefig(os.path.join(os.path.dirname(data_fpath),'impedance.png'),format='png')

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
ax.plot(f,alpha)
ax.set_ylabel(r'$Absorption, \ \alpha$')
ax.set_xlim([500,5e3])
ax.set_ylim([0,1])
ax.grid()
# plt.savefig(os.path.join(os.path.dirname(data_fpath),'absorption.png'),format='png')
