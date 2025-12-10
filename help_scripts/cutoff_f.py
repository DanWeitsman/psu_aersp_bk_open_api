import numpy as np

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
