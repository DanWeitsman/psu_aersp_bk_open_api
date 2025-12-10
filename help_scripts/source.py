
import numpy as np
from scipy.io import wavfile 
from scipy.fft import fft,ifft
import os
from scipy.signal import welch
import matplotlib.pyplot as plt
from scipy.signal import butter,freqs,sosfilt,lfilter,freqz,freqz_sos
#%%

def LFM(f1,f2,Tp,T,A,fs):
    '''
    This function generates an LFM pulse.
    :param f1: starting frequency [Hz]
    :param f2:  ending frequency [Hz]
    :param Tp: duration of the pulse
    :param T:  duration of the entire time series
    :return:
    '''

    dt = fs ** -1
    tp = np.arange(fs * Tp) * dt
    t = np.arange(fs * T) * dt
    phi = np.zeros(int(fs*T))
    phi[:int(fs*Tp)] = 2 * np.pi * (f1 * tp + 1 / 2 * (f2 - f1) * tp ** 2 / Tp)
    xn = A*np.sin(phi)

    return xn

def white_noise( T = 10,A = 0.1,fs = 44100,fc =(),**kwargs):
    '''
    This function returns a time series of white noise with N number of points
    :param T: Total signal duration
    :param A: Amplitude [WU]
    :param fs: Sampling Rate [Hz]
    :param fc: Bandpass cutoff frequencies [Hz]
    :return:
    '''

    N = fs*T
    df = fs/N

    if len(fc)==0:
        filt_ind = np.asarray([1,N/2-1]).astype(int)
    else:
        filt_ind = (np.asarray(fc)/df).astype(int)

    Xm = np.zeros(int(N), dtype=complex)
    phase = np.random.rand(int((np.diff(filt_ind)+1)[0])) * 2 * np.pi

    Xm[filt_ind[0]:filt_ind[1]+1] =  np.sqrt(A*T/2) * np.exp(1j * phase)

    if filt_ind[0] ==1:
        Xm[-filt_ind[1]:] = np.conjugate(Xm[filt_ind[0]:filt_ind[1]+1])[::-1]
    else:
        Xm[-filt_ind[1]:-filt_ind[0]+1] = np.conjugate(Xm[filt_ind[0]:filt_ind[1]+1])[::-1]

    xn = np.real(ifft(Xm)*fs)

    # Xm_2 = np.zeros(int(N), dtype=complex)
    # phase = np.random.rand(int(N/2)-1) * 2 * np.pi

    # Xm[1:int(N / 2)] = A * np.exp(1j * phase)
    # # Xm[(np.arange(1,5)*1000/((fs**-1*N)**-1)).astype(int)] = 10*0.5**(np.arange(1,5))*Xm[(np.arange(1,5)*1000/((fs**-1*N)**-1)).astype(int)]
    # Xm[int(N / 2) + 1:] = A * np.flip(np.conjugate(np.exp(1j * phase)))

    return xn

def pink_noise(T = 10,A = 0.1,fs = 44100,fc =()):
    '''
    This function returns a time series of white noise with N number of points
    :param T: Total signal duration
    :param A: Amplitude [WU]
    :param fs: Sampling Rate [Hz]
    :return:
    '''

    N = fs*T
    df = fs/N

    if len(fc)==0:
        filt_ind = np.asarray([1,N/2-1]).astype(int)
    else:
        filt_ind = (np.asarray(fc)/df).astype(int)

    Xm = np.zeros(int(N), dtype=complex)
    phase = np.random.rand(int((np.diff(filt_ind)+1)[0])) * 2 * np.pi
    f = np.arange(int((np.diff(filt_ind)+1)[0]))*df+filt_ind[0]*df

    Xm[filt_ind[0]:filt_ind[1]+1] =  A/np.sqrt(f) * np.exp(1j * phase)
    
    if filt_ind[0] ==1:
        Xm[-filt_ind[1]:] = np.conjugate(Xm[filt_ind[0]:filt_ind[1]+1])[::-1]
    else:
        Xm[-filt_ind[1]:-filt_ind[0]+1] = np.conjugate(Xm[filt_ind[0]:filt_ind[1]+1])[::-1]

    xn = np.real(ifft(Xm)*fs)

    return xn

def sine(f,T,A,fs):
    '''
    This function generates a sine wave.
    :param f1: starting frequency [Hz]
    :param f2:  ending frequency [Hz]
    :param Tp: duration of the pulse
    :param T:  duration of the entire time series
    :return:
    '''

    N = fs*T
    xn = A * np.sin(2*np.pi*f*np.arange(int(N))*fs**-1)
    return xn

def noise_from_psd(psd,A = 0.1,fs = 44100,fc =()):
    '''
    This function returns a time series of perfect-random noise (only phase is pseudo-random as its sampled from a Gaussian distribution, while magnitude is constant) with a predefined power spectral density
    :param psd: Power spectral density of the signal
    :param T: Total signal duration
    :param A: Amplitude [WU]
    :param fs: Sampling Rate [Hz]
    :return:
    '''

    N = len(psd)
    df = fs/N

    if len(fc)==0:
        filt_ind = np.asarray([1,N/2-1]).astype(int)
    else:
        filt_ind = (np.asarray(fc)/df).astype(int)

    Xm = np.zeros(int(N), dtype=complex)
    phase = np.random.rand(int((np.diff(filt_ind)+1)[0])) * 2 * np.pi
    f = np.arange(int((np.diff(filt_ind)+1)[0]))*df+filt_ind[0]*df

    Xm[filt_ind[0]:filt_ind[1]+1] =  A/np.sqrt(f) * np.exp(1j * phase)
    
    if filt_ind[0] ==1:
        Xm[-filt_ind[1]:] = np.conjugate(Xm[filt_ind[0]:filt_ind[1]+1])[::-1]
    else:
        Xm[-filt_ind[1]:-filt_ind[0]+1] = np.conjugate(Xm[filt_ind[0]:filt_ind[1]+1])[::-1]

    xn = np.real(ifft(Xm)*fs)

    return xn
#%%

fs = 44.1e3
T = 60
A = 0.001
fc = (100,4500)

#%%

xn = white_noise(T,A,fs,fc = fc)

df = (1/T)
nperseg = (df*fs**-1)**-1
f,G= welch(xn, fs=fs, window='boxcar', nperseg=nperseg, noverlap=int(0*nperseg), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')

Xm = fft(xn)*fs**-1
Sxx =  (nperseg/fs)**-1*np.abs(Xm)**2
Gxx = 2*Sxx[:int(nperseg/2)+1]

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
plt.subplots_adjust(left = .15,bottom = .15)
# ax.plot(f,10*np.log10(G*df))
ax.plot(f,G)
ax.plot(f,Gxx)

ax.set_ylabel('SPL [Pa]')
ax.set_xlabel('Frequency [Hz]')
ax.set_xlim([0,5e3])
# ax.set_xscale('log')
# ax.set_ylim([0,1])
ax.grid()


wavfile.write(os.path.join(os.path.dirname(__file__),"WN.wav"), int(fs), xn)

xn = LFM(10,10e3,T/2,T,A,fs)
wavfile.write(os.path.join(os.path.dirname(__file__),"LFM.wav"), int(fs), xn)


xn = pink_noise(T,A,fs,fc = fc)

# df = 1
# nperseg = (df*fs**-1)**-1
# f,G= welch(xn, fs=fs, window='hann', nperseg=nperseg, noverlap=int(0*nperseg), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')


# fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
# plt.subplots_adjust(left = .15,bottom = .15)
# ax.plot(f,10*np.log10(G*df))
# ax.set_ylabel('SPL [Pa]')
# ax.set_xlabel('Frequency [Hz]')
# ax.set_xscale('log')
# ax.grid()

wavfile.write(os.path.join(os.path.dirname(__file__),"PN.wav"), int(fs), xn)
