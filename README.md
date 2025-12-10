# Modified B&K open API scripts for use with the PSU AERSP department's normal incidence impedance tube
This repository contains a modified version of [B&K Open API example scripts](https://github.com/hbkworld/open-api-tutorials.git). The primary changes are:
* A command line interface has been developed for initiating data acquisition and interfaceing with the LAN-XI modules.
* The data is written out to an HDF5 file. 
* A suite of post processing scripts is provided to processing measurements made in Penn State's Aerospace Engineering department's normal-incidence impedance tube (NIT).

## Dependencies

The following packages are required and can be installed through your package manager, pip, or an anaconda distribution.

* [h5py](https://docs.h5py.org/en/latest/build.html)
* Numpy
* Matplotlib 
* Scipy 

This repo was verified to work with Python 3.10, 3.11, and 3.12. 

## Example Usage

To initialize the modules and acquire 10s of data, navigate to the src directory and run the record.py script followed by the ip address of the LAN-XI module. To record data from any other location on your machine it is recommended that the "src" and "help_scripts" directories be added to you .bashrc script. 

The remaining argument options can be displayed by including the -h argument. This allows the user to specify the name of the output file, the sampling rate, the duration of data acquisition, and microphone sensitivity values if they differ from the TED info.

This script can also be used when calibrating the microphones. The data is typically saved out in pascals, however, if the -c argument is included the raw voltages are saved instead. 

The pressure time series and spectra can be plotted if the -p argument is supplied. 

```
    ./record.py "module ip address"
```

## Post Processing

The help_scripts directory contains a suite of post-processing scrips to use with the NIT. Almost all of these scripts should be ran from the command line with the appropriate arguments. 

* generat_source_signal.py can be used to generate white and pink noise signals as well as linear frequency modulated signals. 

* spl.py computes the unweighted sound pressure level at each microphone. This is useful for adjusting the level of the source signal on your computer or the amplifier. 

* spl_at_sample.py educes what the sound pressure level actually is at the surface of the sample.    

* compute_response.py is used to compute the impedance and absorption coefficient of the sample. This does NOT correct for any differences in the response of the microphones which can be eliminated by performing a switch calibration. 

* mic_calibration.py is used to compute the sensitivties of each microphone. It is best to calibrate the microphones one at a time so unplug the other microphone that is not being calibrated from the LAN-XI module. The sensitivity values are written out to a json file and can then be provided to the record.py script via the command line interface.  


