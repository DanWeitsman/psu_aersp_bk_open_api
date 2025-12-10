import h5py
import numpy as np
from scipy.signal import welch,csd,get_window,butter,lfilter
from scipy.fft import fft,ifft
from numpy.lib.stride_tricks import sliding_window_view


#%% ############ File I/O ##################

def read_data_h5(file):
    data = {}
    with h5py.File(file, "r") as f:
        for k, v in f.items():
            if len(v)>1:
                data_temp = {}
                for k_2, v_2 in v.items():
                    data_temp = {**data_temp, **{k_2: v_2[()]}}
                data = {**data,**{k:data_temp}}
            else:
                data = {**data, **{k: v[()]}}
    return data

def read_cal_h5(file):
    cal_data ={}
    with h5py.File(file, "r") as f:
        for k, v in f.items():
            cal_data = {**cal_data, **{k: v[()]}}
    return cal_data

def filt_data(data,fc,fs):
    b, a = butter(4, fc / (fs / 2), 'hp')
    data_filt = lfilter(b, a, data)
    return data_filt

def apply_filt(data,fc,fs):
    for k,v in data.items():
        data[k]['scaled_samples'] = filt_data(v['scaled_samples'],fc,fs)
    return data

#%% ############ Test signal generation ##################

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

    Xm[filt_ind[0]:filt_ind[1]+1] =  A *np.sqrt(T/2)* np.exp(1j * phase)

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

def psd_noise(psd,A = 0.1,fs = 44100,fc =()):
    '''
    This function returns a time series of white noise with N number of points
    :param T: Total signal duration
    :param A: Amplitude [WU]
    :param fs: Sampling Rate [Hz]
    :return:
    '''

    if len(psd)%2:
        N = (len(psd)-1)*2
    else:
        N = len(psd)

    df = fs/N

    if len(fc)==0 or fc[-1]/df>=(N/2-1):
        filt_ind = np.asarray([1,N/2-1]).astype(int)
    else:
        filt_ind = (np.asarray(fc)/df).astype(int)

    Xm = np.zeros(int(N), dtype=complex)
    phase = np.random.rand(int((np.diff(filt_ind)+1)[0])) * 2 * np.pi

    Xm[filt_ind[0]:filt_ind[1]+1] =  np.sqrt(A*1/(2*df)*psd[filt_ind[0]:filt_ind[1]+1]) * np.exp(1j * phase)
    
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

#%% ############ Post Processing ##################

def TF(m1,m2,fs,nperseg,noverlap,window = 'boxcar'):
    G_12 = csd(m1, m2, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')[-1]
    G_11 = welch(m1, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')[-1]
    H = G_12/G_11
    return H

def psd(Sxx):
    nperseg = Sxx.shape[-1]
    Gxx = Sxx[...,:int(nperseg/2)+1]
    if nperseg%2:
        Gxx[...,1:] = 2*Gxx[...,1:]
    else:
        Gxx[...,1:-1] = 2*Gxx[...,1:-1]
    return Gxx

def upsample(xn,fs1,fs2):
    """ This function upsamples a time series to a desired sample rate

    Args:
        xn (array): original time series os shape (N_channels x N_samples)
        fs1 (float): original sampling rate [Hz]
        fs2 (float): desired sampling rate [Hz]

    Returns:
        array: upsampled time series
    """
    N1 = xn.shape[-1]
    N2 = int(N1*fs2/fs1)

    if len(xn.shape)>1:
        Xm_upsampled = np.zeros((xn.shape[0],N2),dtype=complex)
    else:
        Xm_upsampled = np.zeros(N2,dtype=complex)

    Xm = fft(xn,axis = -1)*fs1**-1

    if N1%2:
        Xm_upsampled[...,:int(N1/2)+1] = Xm[...,:int(N1/2)+1]
        Xm_upsampled[...,-int(N1/2):] = Xm[...,-int(N1/2):]

    else:
        Xm_upsampled[...,:int(N1/2)] = Xm[...,:int(N1/2)]
        Xm_upsampled[...,-int(N1/2)+1:] = Xm[...,-int(N1/2)+1:]
        Xm_upsampled[...,int(N1/2)] = Xm[...,int(N1/2)]/2
        Xm_upsampled[...,-int(N1/2)] = Xm[...,int(N1/2)]/2

    xn_upsampled = np.real(ifft(Xm_upsampled,axis = -1)*fs2)

    return xn_upsampled

def acs_response(m1,m2,fs,df,s = 0.0366167,l = .1380267,sos = 343.56,overlap = 0.0,window = 'boxcar'):
    """ This function computes the reflection coefficient of a sample 

    Args:
        m1 (np.array): pressure time series of the mic that is closest to the driver [pa]
        m2 (np.array): pressure time series of the mic that is closest to the sample [pa]
        fs (float): sampling rate [Hz]
        df (float): desired frequency resolution
        s (float, optional): microphone spacing [m]. Defaults to 0.0366167.
        l (float, optional): distance between the sample and the mic that is closer to the driver [m]. Defaults to .1380267.
        sos (float, optional): speed of sound [m/s]. Defaults to 343.56.
        overlap (float, optional): percentage overlap between records. Defaults to 0.0.
        window (str, optional): window function. Defaults to 'boxcar'.

    Returns:
        R (np.array): complex reflection coefficient of the sample
        Z (np.array): specific acoustic impedance of a sample
        alpha (np.array): absorption coefficient of the sample
    """
    # number of points per record
    nperseg = int(fs/df)
    # number of points to overlap
    noverlap = int(np.round(nperseg*overlap))
    # transfer function between the two microphones
    H_12 = TF(m1,m2,fs = fs,nperseg = nperseg,noverlap = noverlap,window = window)

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

    return R,Z,alpha


def psd_at_sample(m1,m2,fs,df,s = 0.0366167,l = .1380267,sos = 343.56,overlap = 0.0,window = 'boxcar'):
    """ This function computes the PSD on the surface of the sample from the two offset acoustic measurements. It is an extension of the two-microphone method. 
    Like the standard two-microphone method, the assumption of plane-wave propagation still applies so it is only valid
    in that frequency range. It also does not perform well at frequencies that are integer multiples of the quarter wavelength frequency based on the distance from 
    either mics to the test sample. 

    Args:
        m1 (np.array): pressure time series of the mic that is closest to the driver [pa]
        m2 (np.array): pressure time series of the mic that is closest to the sample [pa]
        fs (float): sampling rate [Hz]
        df (float): desired frequency resolution
        s (float, optional): microphone spacing [m]. Defaults to 0.0366167.
        l (float, optional): distance between the sample and the mic that is closer to the driver [m]. Defaults to .1380267.
        sos (float, optional): speed of sound [m/s]. Defaults to 343.56.
        overlap (float, optional): percentage overlap between records. Defaults to 0.0.
        window (str, optional): window function. Defaults to 'boxcar'.

    Returns:
        f (np.array): frequency array [Hz]
        Gxx (np.array): single-sided power spectral density [Pa^2/Hz]
    """
    # number of points per record
    nperseg = int(fs/df)
    # number of points to overlap 
    noverlap = int(np.round(overlap*nperseg))

    # temporal resolution [s]
    dt = 1/fs
    # frequency array [Hz]
    f = np.arange(int(nperseg/2)+1)*df
    # wave number [m^-1]
    k = 2*np.pi*f/sos

    R,_,_ = acs_response(m1,m2,fs,df,s = s,l = l,sos = sos,overlap = overlap,window = window)

    H_1s = ((R*np.exp(-1j*k*l)+np.exp(-1j*k*l))/(1+R*np.exp(-1j*2*k*l)))

    filt_resp = np.ones(nperseg,dtype = complex)
    filt_resp[:int(nperseg/2)+1] = H_1s
    if nperseg%2:
        filt_resp[int(nperseg/2)+1:] = np.conj(H_1s[1:])[::-1]
    else:
        filt_resp[int(nperseg/2)+1:] = np.conj(H_1s[1:-1])[::-1]

    win = get_window(window,nperseg)
    m1_windowed = sliding_window_view(m1, window_shape = nperseg, axis=-1)[::(nperseg-noverlap)]*win/np.sqrt(np.mean(win**2))

    Xm = fft(m1_windowed,axis = -1)*dt
    Xm_samp = Xm*filt_resp
    Sxx_samp = (nperseg*dt)**-1*np.abs(Xm_samp)**2
    Gxx_samp = np.mean(psd(Sxx_samp),axis = 0)

    return f,Gxx_samp

def uniform_spl_filt_response(m1,m2,fs,fs_src = 44.1e3,s = 0.0366167,l = .1380267,sos = 343.56):
    """ This function returns the filter response to apply to the input signal in order to achieve a uniform spl w/frequency at the sample. 

    Args:
        m1 (np.array): pressure time series of the mic that is closest to the driver [pa]
        m2 (np.array): pressure time series of the mic that is closest to the sample [pa]
        fs (float): sampling rate [Hz]
        fs_src (float): sampling rate of system that is used to generate the test signal [Hz]
        s (float, optional): microphone spacing [m]. Defaults to 0.0366167.
        l (float, optional): distance between the sample and the mic that is closer to the driver [m]. Defaults to .1380267.
        sos (float, optional): speed of sound [m/s]. Defaults to 343.56.

    Returns:
        uniform_spl_filt_resp (np.array): psd to use to filter the input signal to achieve a uniform spl w/frequency at the sample 
    """

    # total number of points acquiered
    N1 = m1.shape[-1]
    # total number of points of the test signal
    N2 = int(fs_src*N1/fs)
    # desired sampling rate [Hz]
    fs2 = fs*N2/N1

    # upsampled measured time series
    data_upsample = upsample(np.asarray((m1,m2)),fs1 = fs,fs2 = fs2)
    
    f,Gxx = psd_at_sample(data_upsample[0],data_upsample[1],fs = fs2,df = fs2/N2,s = s,l = l,sos = sos,overlap = 0.0,window = 'boxcar')
    Gxx[np.isnan(Gxx) | (Gxx<1.11e-16)] = 0

    uniform_spl_filt_resp = np.mean(Gxx)/Gxx
    uniform_spl_filt_resp[np.isinf(uniform_spl_filt_resp)] = 0


    return uniform_spl_filt_resp

