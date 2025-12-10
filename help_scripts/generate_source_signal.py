import numpy as np
import matplotlib.pyplot as plt
from help_functions import *
import os
import argparse
from scipy.io import wavfile 

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", dest="name", default="source",required=False, \
        help="Name of file")
    parser.add_argument("-LFM", action="store_true",required=False, \
        help="Include to generate a LFM pulse")
    parser.add_argument("-f", default=(100,10000),nargs=2, type=float,required=False, \
        help="minimum and maximum frequencies of LFM pulse")
    parser.add_argument("-WN", action="store_true",required=False, \
        help="Include to generate a white noise")
    parser.add_argument("-PN", action="store_true",required=False, \
        help="Include to generate pink noise")
    parser.add_argument("-T", default=10, type=float,required=False, \
        help="total duration of signal [sec]")
    parser.add_argument("-A", default=0.01, type=float,required=False, \
        help="amplitude of signal as percentage of your devices full-scale output range")
    parser.add_argument("-fs", default=44.1e3, type=float,required=False, \
        help="your devices' sampling rate, defaults to 44.1kHz")
    parser.add_argument("-fc", default=(200,4500),nargs=2, type=float,required=False, \
        help="lower and upper cutoff frequencies of bandpass filter if desired")
    args = parser.parse_args()

    if args.LFM is not None:
        xn = LFM(args.f[0],args.f[1],args.T/2,args.T,args.A,args.fs)

    if args.WN is not None:
        xn = white_noise(args.T,args.A,args.fs,fc = args.fc)

    if args.PN is not None:
        xn = pink_noise(args.T,args.A,args.fs,fc = args.fc)
    
    wavfile.write(os.path.join(os.getcwd(),f"{args.name}.wav"), int(args.fs), xn)


if __name__ == "__main__":
    main()