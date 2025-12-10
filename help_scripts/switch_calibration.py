import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from help_functions import *
from scipy.signal import welch,csd

#%%


# absolute path of claibration measurement h5 file of the standard mic configuration
std_path = os.path.join(os.path.dirname(__file__),'..','std_cal_aoe1_100db_1','std_cal_aoe1_100db_1.h5')
# absolute path of claibration measurement h5 file of the switched mic configuration
switch_path = os.path.join(os.path.dirname(__file__),'..','switch_cal_aoe1_100db_1','switch_cal_aoe1_100db_1.h5')

std_data, switch_data = list(map(lambda f: read_data_h5(f),[std_path,switch_path]))

#%%

#%%

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

# fc = 40
# std_data, switch_data = list(map(lambda f: apply_filt(f,fc = fc,fs = fs),[std_data,switch_data]))

# 
# H_std, H_switch = list(map(lambda f: TF(f['channel0']['scaled_samples'],f['channel1']['scaled_samples'],fs = fs,nperseg=nperseg,noverlap = noverlap),[std_data,switch_data]))

H_std =  TF(std_data['channel0']['scaled_samples'],std_data['channel1']['scaled_samples'],fs = fs,nperseg=nperseg,noverlap = noverlap)
H_switch =  TF(switch_data['channel1']['scaled_samples'],switch_data['channel0']['scaled_samples'],fs = fs,nperseg=nperseg,noverlap = noverlap)
# H_c = (H_std*H_switch)**(1/2)
H_c=(abs(H_std)*abs(H_switch))**(1/2)*np.exp(1j*0.5*(np.unwrap(np.angle(H_std))+np.unwrap(np.angle(H_switch))))


# std_data_dist = 1.6*np.roll(std_data['channel0']['scaled_samples'],-10)
# switch_data_dist = 1.6*np.roll(switch_data['channel0']['scaled_samples'],-10)

# H_std_dist =  TF(std_data_dist,std_data['channel1']['scaled_samples'],fs = fs,nperseg=nperseg,noverlap = noverlap)
# H_switch_dist =  TF(switch_data['channel1']['scaled_samples'],switch_data_dist,fs = fs,nperseg=nperseg,noverlap = noverlap)
# H_c_dist=(abs(H_std_dist)*abs(H_switch_dist))**(1/2)*np.exp(1j*0.5*(np.unwrap(np.angle(H_std_dist))+np.unwrap(np.angle(H_switch_dist))))


# f = np.arange(len(H_std))*df
# fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
# ax[0].plot(f,abs(H_std))
# ax[0].plot(f,abs(H_std_dist),linestyle = '--')
# ax[0].set_ylabel(r'$H_{12}$')
# ax[0].grid()
# ax[1].plot(f,np.unwrap(np.angle(H_std)))
# ax[1].plot(f,np.unwrap(np.angle(H_std_dist)),linestyle = '--')
# ax[1].set_ylabel(r'$\phi \ [rad]$')
# ax[1].legend(['Original','Distorted'])
# ax[1].set_xlabel('Frequency [Hz]')
# ax[1].grid()

# fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
# ax[0].plot(f,abs(H_switch))
# ax[0].plot(f,abs(H_switch_dist),linestyle = '--')
# ax[0].set_ylabel(r'$H_{12}$')
# ax[0].grid()
# ax[1].plot(f,np.unwrap(np.angle(H_switch)))
# ax[1].plot(f,np.unwrap(np.angle(H_switch_dist)),linestyle = '--')
# ax[1].set_ylabel(r'$\phi \ [rad]$')
# ax[1].legend(['Original','Distorted'])
# ax[1].set_xlabel('Frequency [Hz]')
# ax[1].grid()

# fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
# ax[0].plot(f,abs(H_c))
# ax[0].plot(f,abs(H_c_dist),linestyle = '--')
# ax[0].set_ylabel(r'$H_{c}$')
# ax[0].grid()
# ax[1].plot(f,np.unwrap(np.angle(H_c)))
# ax[1].plot(f,np.unwrap(np.angle(H_c_dist)),linestyle = '--')
# ax[1].set_ylabel(r'$\phi \ [rad]$')
# ax[1].legend(['Original','Distorted'])
# ax[1].set_xlabel('Frequency [Hz]')
# ax[1].grid()


# fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
# ax[0].plot(f,abs(H_std))
# ax[0].plot(f,abs(H_c),linestyle = '--')
# ax[0].set_ylabel(r'$H_{c}$')
# ax[0].grid()
# ax[1].plot(f,np.angle(H_std))
# ax[1].plot(f,np.angle(H_c),linestyle = '--')
# ax[1].set_ylabel(r'$\phi \ [rad]$')
# ax[1].legend(['Original','Distorted'])
# ax[1].set_xlabel('Frequency [Hz]')
# ax[1].grid()


#%%

save_dir  =os.path.join(os.path.dirname(__file__),'switch_cal.h5')
if os.path.exists(save_dir):
    os.remove(save_dir)

data_out = {'cal_correction':H_c,'fs':fs,'df':df,'nperseg':nperseg,'noverlap':noverlap}

with h5py.File(save_dir, 'a') as f:
    for k, v in data_out.items():
        f.create_dataset(k, shape=np.shape(v), data=v)
