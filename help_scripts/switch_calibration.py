import numpy as np
import os
import h5py
from help_functions import *
import matplotlib.pyplot as plt
import plot_styles 

#%%

# absolute path of claibration measurement h5 file of the standard mic configuration
std_path = os.path.join(os.path.dirname(__file__),'..','std_cal_aoe1_100db_1','std_cal_aoe1_100db_1.h5')
# absolute path of claibration measurement h5 file of the switched mic configuration
switch_path = os.path.join(os.path.dirname(__file__),'..','switch_cal_aoe1_100db_1','switch_cal_aoe1_100db_1.h5')

std_data, switch_data = list(map(lambda f: read_data_h5(f),[std_path,switch_path]))

# speed of sound [m/s]
sos = 343.563
# distance between the sample and the mic that is closer to the driver [m]
l = .1380267
#microphone spacing [m]
s = 0.0366167
# frequency resolution [Hz]
df = 2
# sampling rate [Hz]
fs = std_data[list(std_data.keys())[0]]['sample_rate']
# temporal resolution [s]
dt = fs**-1
# samples per record
nperseg = (df*dt)**-1
# percentage overlap between records
noverlap = 0.5

H_std =  TF(std_data['channel0']['scaled_samples'],std_data['channel1']['scaled_samples'],fs = fs,nperseg=nperseg,noverlap = noverlap)
H_switch =  TF(switch_data['channel1']['scaled_samples'],switch_data['channel0']['scaled_samples'],fs = fs,nperseg=nperseg,noverlap = noverlap)
# H_c = (H_std*H_switch)**(1/2)
H_12=(abs(H_std)*abs(H_switch))**(1/2)*np.exp(1j*0.5*(np.unwrap(np.angle(H_std))+np.unwrap(np.angle(H_switch))))

# frequency array [Hz]
f = np.arange(int(nperseg/2)+1)*df
# wave number [m^-1]
k = 2*np.pi*f/sos

# acoustic transfer function of the incident acoustic wave between the two microphones (assuming plane wave propagation without thermoviscous lossess)
H_i = np.exp(-1j*k*s)
# acoustic transfer function of the reflected acoustic wave between the two microphones
H_r = np.exp(1j*k*s)

# Reflection coefficient at the first mic (closer to the driver)
R_1 = (H_12-H_i)/(H_r-H_12)
# Reflection coefficient of the test sample
R = R_1*np.exp(1j*2*k*l)

# Acoustic impedance of the test sample
Z = (1+R)/(1-R)
# absorption coefficient of the test sample
alpha =  1 - abs((Z-1)/(Z+1))**2

fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
plt.subplots_adjust(left = 0.15,bottom=0.15,hspace = 0.3)
ax[0].plot(f,np.real(Z))
ax[1].plot(f,np.imag(Z))
ax[0].set(xticklabels = [],ylabel= r'$Resistance, \ \overline{\theta}$',xlim =[0,4e3],ylim = [-10,10])
ax[1].set(ylabel= r'$Reactance, \ \overline{\chi}$',xlabel = r'$Frequency \ [Hz]$',xlim =[0,4e3],ylim = [-10,10])
ax[0].grid()
ax[1].grid()

fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
plt.subplots_adjust(left = 0.15,bottom=0.15,hspace = 0.3)
ax[0].plot(f,np.abs(R))
ax[1].plot(f,alpha)
ax[0].set(xticklabels = [],ylabel= r'$Reflection, \ |\mathit{R}|$',xlim =[100,4e3],ylim = [0,1])
ax[1].set(ylabel= r'$Absorption, \ \alpha$',xlabel = r'$Frequency \ [Hz]$',xlim =[100,4e3],ylim = [0,1])
ax[0].grid()
ax[1].grid()


