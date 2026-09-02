# Penn State AERSP NIT Open API Data Acquisition Suite

A comprehensive data acquisition and post-processing system for the Penn State Aerospace Engineering Department's Normal-Incidence Impedance Tube (NIT). This repository extends [Brüel & Kjær's Open API example scripts](https://github.com/hbkworld/open-api-tutorials.git) with a command-line interface for real-time acoustic measurements and material characterization.

## Project Overview

This suite provides:
- **Real-time data acquisition**: Command-line interface for the LAN-XI modules with configurable sampling rates and duration
- **HDF5 data storage**: Efficient data format for time series and frequency domain measurements
- **Post-processing tools**: Comprehensive suite of scripts for acoustic analysis and material characterization
- **Material characterization**: Compute impedance and absorption coefficients for acoustic materials

## Requirements

### Python Version
- Python 3.10, 3.11, or 3.12 (verified compatible)

### Dependencies
Install required packages using pip or your package manager:

```bash
pip install -r requirements.txt
```

Key dependencies:
- **[h5py](https://docs.h5py.org/en/latest/build.html)** - HDF5 file format support
- **NumPy** - Numerical computations
- **SciPy** - Scientific computing and signal processing
- **Matplotlib** - Data visualization and plotting

### Hardware Requirements
- Brüel & Kjær LAN-XI module(s) with network connectivity
- Microphones with appropriate sensitivity for your application
- DAQ interface compatible with B&K Open API

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd psu_aersp_bk_open_api
```

### 2. Create Virtual Environment (Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add to PATH (Optional)
For easier access from any directory, add the src and help_scripts directories to your `$PATH`:

```bash
export PATH="$PATH:/path/to/psu_aersp_bk_open_api/src:/path/to/psu_aersp_bk_open_api/help_scripts"
```

Add this line to your `.bashrc` or `.zshrc` for persistence.

## Data Acquisition

### Basic Usage

To acquire 10 seconds of acoustic data from a LAN-XI module:

```bash
cd src
./record.py <module-ip-address>
```

Replace `<module-ip-address>` with the network address of your LAN-XI module (e.g., `192.168.1.100`).

### Command-Line Options

View all available options:

```bash
./record.py -h
```

Common options include:
- `-o, --output`: Specify output HDF5 filename
- `-r, --rate`: Sampling rate in Hz (default: 51200 Hz)
- `-d, --duration`: Duration of data acquisition in seconds (default: 10s)
- `-m, --microphone`: Microphone sensitivity values (IEPE or TED configuration)
- `-c, --calibration`: Save raw voltages instead of pressure (for calibration purposes)
- `-p, --plot`: Generate plots of pressure time series and frequency spectra after acquisition

### Examples

Acquire 30 seconds of data at 44.1 kHz and plot results:
```bash
./record.py 192.168.1.100 -d 30 -r 44100 -p -o measurement_001.h5
```

Acquire raw voltage data for microphone calibration:
```bash
./record.py 192.168.1.100 -c -o calibration_raw.h5
```

## Post-Processing Tools

The `help_scripts` directory contains utility scripts for analyzing acoustic measurements and material properties. Most scripts accept command-line arguments—use the `-h` flag to display available options.

### Signal Generation

**generate_source_signal.py**
- Generates acoustic test signals for NIT measurements
- Supported signal types:
  - White noise (flat frequency spectrum)
  - Pink noise (1/f spectrum)
  - Linear frequency modulated (LFM) chirps
- Useful for sweeping frequency ranges or stochastic testing

### Acoustic Analysis

**spl.py**
- Computes unweighted sound pressure level (SPL) at each microphone
- Useful for:
  - Verifying measurement levels during setup
  - Adjusting source signal amplification
  - Quality assurance checks

**spl_at_sample.py**
- Estimates sound pressure level at the sample surface
- Accounts for propagation from microphone positions
- Helps verify adequate acoustic loading on the sample

### Material Characterization

**compute_response.py**
- Calculates acoustic impedance and absorption coefficient
- **Note**: Does not correct for microphone response mismatch
- To eliminate microphone differences, perform a switch calibration first (see below)
- Output includes both impedance and absorption vs. frequency

**mic_calibration.py**
- Determines the sensitivity of each microphone in V/Pa
- Best practice: Calibrate microphones one at a time
  - Disconnect the microphone not being calibrated from the LAN-XI module
  - Connect calibrated microphone to a reference signal source
- Outputs sensitivity values to `mic_sens.json`
- Use calibration results with the `record.py` script via the `-m` option

### Utility Scripts

**mic_sens.json**
- JSON configuration file for microphone sensitivity values
- Updated by `mic_calibration.py`
- Referenced by `record.py` for accurate pressure measurements

**help_functions.py**
- Shared utility functions used by other scripts
- Includes filtering, windowing, and common DSP operations

**plot_styles.py**
- Matplotlib configuration for consistent visualization across all scripts

## Directory Structure

```
psu_aersp_bk_open_api/
├── src/
│   ├── record.py                    # Main data acquisition script
│   ├── streaming_interpretation.py  # Real-time data processing
│   ├── plot.py                      # Plotting utilities
│   └── kaitai/                      # Kaitai Struct definitions for API messages
├── help_scripts/
│   ├── generate_source_signal.py    # Test signal generation
│   ├── spl.py                       # Sound pressure level computation
│   ├── spl_at_sample.py             # Sample surface SPL estimation
│   ├── compute_response.py          # Impedance & absorption calculation
│   ├── mic_calibration.py           # Microphone sensitivity calibration
│   ├── mic_sens.json                # Microphone calibration data
│   └── [utility scripts]            # Helper functions and utilities
├── requirements.txt                 # Python package dependencies
└── README.md                        # This file
```

