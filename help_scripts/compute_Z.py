import numpy as np
import matplotlib.pyplot as plt
import os
from help_functions import *

#%%

# absolute path to h5 file containing acoustic data
data_fpath = '/Users/danielweitsman/codes/github/hbkworld/open-api-tutorials/nit_tutorial_test/9_5_25/pnt_2/pnt_2.h5'

data_fpath = os.path.join(os.path.dirname(__file__),'..','cal_switched_2','cal_switched_2.h5')

# absolute path to h5 file containing switched calibration correction  
cal_fpath = os.path.join(os.path.dirname(__file__),'switch_cal.h5')

# microphone spacing [m]
s = 0.0366167
# distance to the sample [m]
l = .10141

L = 0.057

# sos [m/s]
c = 20.047*np.sqrt(273.15+20.556)
# density [kg/m^3]
rho = 1.125

#%%

data = read_data_h5(data_fpath)
# cal_data = read_cal_h5(cal_fpath)

#%%
# fc = 40
# data =  apply_filt(data,fc  = fc,fs = cal_data['fs'])

#%%
N = len(cal_data['cal_correction'])
fs = data['channel0']['sample_rate']
# frequency vector [Hz]
f = np.arange(int(N))*cal_data['df']
# wave number 
k = 2*np.pi*f/c

# computes transfer function between the two mics
H_12 = TF(data['channel0']['scaled_samples'],data['channel1']['scaled_samples'],fs = cal_data['fs'],nperseg = cal_data['nperseg'],noverlap = cal_data['noverlap'])

H =H_12


# H = np.abs(H_12)/np.abs(cal_data['cal_correction'])*np.exp(1j*(np.angle(H_12)-np.angle(cal_data['cal_correction'])))
# H = 1/np.abs(cal_data['cal_correction'])*(np.real(H_12)*np.cos(np.angle(cal_data['cal_correction']))+np.imag(H_12)*np.sin(np.angle(cal_data['cal_correction'])))+1j*1/np.abs(cal_data['cal_correction'])*(np.imag(H_12)*np.cos(np.angle(cal_data['cal_correction']))-np.real(H_12)*np.sin(np.angle(cal_data['cal_correction'])))


#%%
# acoustic transfer function of the incident acoustic wave between the two micriophones
H_i = np.exp(-1j*k*s)
# acoustic transfer function of the reflected acoustic wave between the two micriophones
H_r = np.exp(1j*k*s)

# Reflection coefficient at the first mic (closer to the driver)
R_1 = (H-H_i)/(H_r-H)
# Reflection coefficient of the test sample
R = R_1*np.exp(1j*2*k*(l+s))


# D = 1+np.real(H)**2+np.imag(H)**2-2*(np.real(H)*np.cos(k*s)+np.imag(H)*np.sin(k*s))
# R2 = (2*np.real(H)*np.cos(k*(2*l+s))-np.cos(2*k*l)-(np.real(H)**2+np.imag(H)**2)*np.cos(2*k*(l+s)))/D+1j*(2*np.real(H)*np.sin(k*(2*l+s))-np.sin(2*k*l)-(np.real(H)**2+np.imag(H)**2)*np.sin(2*k*(l+s)))/D

# R_mag = ((1+np.abs(H)**2-2*np.abs(H)*np.cos(np.angle(H)+k*s))/(1+np.abs(H)**2-2*np.abs(H)*np.cos(np.angle(H)-k*s)))**(1/2)
# phi_2 = 2*k*(l+s)+np.arctan((2*np.abs(H)*np.cos(np.angle(H))*np.sin(k*s)-np.sin(2*k*s))/(np.abs(H)**2-2*np.abs(H)*np.cos(np.angle(H))*np.cos(k*s)+np.cos(2*k*s)))
# R2 = R_mag*np.exp(1j*phi_2)

# Acoustic impedance of the test sample
Z = (1+R)/(1-R)
# absorption coefficient of the test sample
alpha =  1 - abs((Z-1)/(Z+1))**2
# alpha = 4* np.real(Z)/((np.real(Z)+1)**2+np.imag(Z)**2)

#%%

H =cal_data['cal_correction']


# H = np.abs(H_12)/np.abs(cal_data['cal_correction'])*np.exp(1j*(np.angle(H_12)-np.angle(cal_data['cal_correction'])))
# H = 1/np.abs(cal_data['cal_correction'])*(np.real(H_12)*np.cos(np.angle(cal_data['cal_correction']))+np.imag(H_12)*np.sin(np.angle(cal_data['cal_correction'])))+1j*1/np.abs(cal_data['cal_correction'])*(np.imag(H_12)*np.cos(np.angle(cal_data['cal_correction']))-np.real(H_12)*np.sin(np.angle(cal_data['cal_correction'])))



#%%

fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
ax[0].tick_params(axis = 'x', labelsize=0)
ax[0].plot(f,np.real(Z))
ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
ax[0].set_xlim([0,5e3])
ax[0].set_ylim([-10,10])
ax[0].grid()

ax[1].plot(f,np.imag(Z))
# ax[1].plot(f,-(np.tan(k*L))**-1,linestyle = '-.')
ax[1].set_ylabel(r'$Reactance, \ \overline{\chi}$')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_xlim([0,5e3])
ax[1].set_ylim([-10,10])
# ax[1].legend(['Original','Corrected','Analytical'])
ax[1].grid()
plt.savefig(os.path.join(os.path.dirname(data_fpath),'impedance.png'),format='png')

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
ax.plot(f,alpha)
ax.set_ylabel(r'$Absorption, \ \alpha$')
ax.set_xlim([0,5e3])
ax.set_ylim([0,1])
ax.grid()
# ax.legend(['Original','Corrected'])

plt.savefig(os.path.join(os.path.dirname(data_fpath),'absorption.png'),format='png')


