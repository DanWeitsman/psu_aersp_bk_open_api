import numpy as np
import matplotlib.pyplot as plt

#%%

# height of cross section [m]
a = 2/39.37
# width of cross section [m]
b = 2/39.37
# length of impedance tube [m]
L = 35.9843/39.37

# microphone separation distance [m]
s = 1.25/39.37
# distance between the sample and the closest mic
l = 2.5/39.37
SPL = 100

fs = 40e3
df = 5

# speed of sound [m/s]
sos = 343

# number of requested modes
N_modes = 2

#%%
# mode number
mode = np.arange(N_modes)+1
n,m = np.meshgrid(mode,mode)
# cutoff frequency
fc = 0.5*sos*np.sqrt((m/a)**2+(n/b)**2)

#%%

dt = fs**-1
N = (dt*df)**-1
p0 = 10**(SPL/20)*20e-6

t = np.arange(N)*dt
f = np.arange(N)*df
w = 2*np.pi*f
k = w/sos

z = np.array([[L-l,L-(l+s)]]).T

p = p0*np.exp(1j*(w*t-k*z))

S_12 = np.conj(p[1])*p[0]
S_11 =  np.abs(p[1])**2
H_12 = S_12/S_11
H_i = np.exp(-1j*k*s)
H_r = np.exp(1j*k*s)

R_1 = (H_12-H_i)/(H_r-H_12)
R = R_1*np.exp(1j*2*k*l)
Z = ((1+R)/(1-R))

H_12 = p[1]/p[0]
# acoustic transfer function of the incident acoustic wave between the two micriophones
H_i = np.exp(-1j*k*s)
# acoustic transfer function of the reflected acoustic wave between the two micriophones
H_r = np.exp(1j*k*s)

# Reflection coefficient at the first mic (closer to the driver)
R_1 = (H_12-H_i)/(H_r-H_12)
# Reflection coefficient of the test sample
R = R_1*np.exp(1j*2*k*l)
# Acoustic impedance of the test sample
Z = (1+R)/(1-R)

#%%

fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.15)
ax[0].tick_params(axis = 'x', labelsize=0)
ax[0].plot(f,np.real(Z))
ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
ax[0].set_xlim([1400,2200])
# ax[0].set_ylim([0,.15])
ax[0].grid()

ax[1].plot(f,np.imag(Z))
ax[1].set_ylabel(r'$Reactance, \ \overline{\chi}$')
ax[1].set_xlim([0,5000])
# ax[1].set_ylim([-5,5])
ax[1].grid()

#%%
fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
levels = np.linspace(0, np.max(fc), 25)
c = ax.contourf(n, m, fc,cmap = 'hot',levels = levels)
cbar = fig.colorbar(c,pad = .075)
ax.set_ylabel('m')
ax.set_xlabel('n')
cbar.set_label('Frequency [Hz]')
