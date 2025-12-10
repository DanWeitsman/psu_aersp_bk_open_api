#!/usr/bin/env python3
# pylint: disable=C0103

"""Example code to demonstrate single-module streaming with the LAN-XI Open API."""

import time
import socket
import threading
import selectors
import numpy as np
import os
import requests
import json

def receive_thread(sel):
    """On selector event, reads data from associated socket and writes to file."""
    while True:
        events = sel.select()
        for key, _ in events:
            sock = key.fileobj
            data = sock.recv(16384)
            if not data:
                return
            file = key.data
            file.write(data)

def get_measurement_errors(cs):
    """Given a 'cs' array of channelStatus structures, return a collection of error strings."""
    err = []
    for i, sts in enumerate(cs):
        s, prefix = "", f"Channel {i+1}: "
        if sts is None:
            continue
        if sts["anol"] != "none":
            s = s + prefix + f'Analog Overload ({sts["anol"]})'
            prefix = ", "
        if sts["cmol"] != "none":
            s = s + prefix + f'Common Mode Overload ({sts["cmol"]})'
            prefix = ", "
        if sts["cf"] != "none":
            s = s + prefix + f'Cable Fault ({sts["cf"]})'
            prefix = ", "
        if s != "":
            err.append(s)
    return err

# parser = argparse.ArgumentParser()
# parser.add_argument("addr", help="IP address of the LAN-XI module")
# parser.add_argument("-n", "--name", dest="name", default="My Measurement", \
#     help="Name of the measurement")
# parser.add_argument("-d", "--save_dir", dest="save_dir", default="./", \
#     help="Absolute path of where to save the data")
# parser.add_argument("-fs", "--sampling_rate", default=20e3, type=float, \
#     help="The sampling rate (in Hz) of the measurement")
# parser.add_argument("-t", "--time", dest="time", default=10, type=int, \
#     help="The time (in seconds) of the measurement")
# parser.add_argument("-s", "--scale", action="store_true", \
#     help="Include this flag to scale the measurements by a different scale factor (inputRange x headroom / sensitivity) than what is stored in the TEDS data. If a json file with the microphone sesitivity is avaliable it will use the sensitivity value from there.")
# kwargs = parser.parse_kwargs()

def streaming_single_module(**kwargs):

    # Generate base URL; IPv6 addresses in URL's must be enclosed in square brackets
    ip_addr = kwargs['addr'].split("%")[0] # Remove IPv6 zone index, if specified
    addr_family = socket.getaddrinfo(ip_addr, port=0)[0][0]
    base_url = "http://[" + kwargs['addr'] + "]" if addr_family == socket.AF_INET6 else "http://" + kwargs['addr']
    base_url = base_url + "/rest/rec"
    kwargs['addr'] = kwargs['addr'].replace("%%", "%") # Fix double per cent sign issue on Windows

    requests.put(base_url + "/open")


    print(f"Creating {kwargs['time']}-second measurement {kwargs['name']} on module at {ip_addr}")

    # Set the module date/time
    now = time.time()
    requests.put(base_url + "/module/time", data=str(int(now * 1000)))

    #
    module_info = requests.get(base_url + "/module/info").json()
    supported_sample_rates = sorted(module_info["supportedSampleRates"])
    # supported_input_ranges = sorted(module_info["supportedRanges"])
    # supported_filters = sorted(module_info["supportedFilters"])

    # Determines which supported sample rate is closest to the specified.
    fs = supported_sample_rates[abs(np.asarray(supported_sample_rates)-kwargs['sampling_rate']).argmin()]
    # Computes the corresponding bandwidth and converts it to a str.
    BW = f'{fs/2.56/1e3} kHz'

    # Open the recorder and enter the configuration state
    requests.put(base_url + "/open", json={"performTransducerDetection": False})
    requests.put(base_url + "/create")

    # Start TEDS transducer detection
    print("Detecting transducers...")
    requests.post(base_url + "/channels/input/all/transducers/detect")

    # Wait for transducer detection to complete
    prev_tag = 0
    while True:
        response = requests.get(base_url + "/onchange?last=" + str(prev_tag)).json()
        prev_tag = response["lastUpdateTag"]
        if not response["transducerDetectionActive"]:
            break

    # Get the result of the detection
    transducers = requests.get(base_url + "/channels/input/all/transducers").json()

    # Get the default measurement setup
    setup = requests.get(base_url + "/channels/input/default").json()

    # Select streaming rather than (the default) recording to SD card
    for ch in setup["channels"]:
        ch["destinations"] = ["socket"]

    # Configure front-end based on the result of transducer detection
    for idx, t in enumerate(transducers):
        if t is not None:
            setup["channels"][idx]["bandwidth"] = BW
            setup["channels"][idx]["filter"] = '22.4 Hz'
            setup["channels"][idx]["transducer"] = t
            setup["channels"][idx]["ccld"] = t["requiresCcld"]
            setup["channels"][idx]["polvolt"] = t["requires200V"]
            print(f'Channel {idx+1}: {t["type"]["number"] + " s/n " + str(t["serialNumber"])}, '
                f'CCLD {"On" if t["requiresCcld"] == 1 else "Off"}, '
                f'Polarization Voltage {"on" if t["requires200V"] == 1 else "off"}')
            if kwargs['sensitivity']:
                setup["channels"][idx]["sensitivity"] = kwargs['sensitivity'][idx]

    # removes unused channels
    setup["channels"] = [ch for idx, ch in enumerate(setup["channels"]) if transducers[idx]!=None]

    with open(os.path.join(kwargs['save_dir'],'config.json'),'w') as f:
        json.dump(setup,f,indent=3)

    # Apply the setup
    print(f'Configuring module...')
    requests.put(base_url + "/channels/input", json=setup)

    # Store streamed data to this file
    stream_file = open(os.path.join(kwargs['save_dir'],kwargs['file']), "wb")

    # Request the port number that the module is listening on
    port = requests.get(base_url + "/destination/socket").json()["tcpPort"]

    # Connect streaming socket
    stream_sock = socket.create_connection((kwargs['addr'], port), timeout=10)
    stream_sock.setblocking(False)

    # We'll use a Python selector to manage socket I/O
    selector = selectors.DefaultSelector()
    selector.register(stream_sock, selectors.EVENT_READ, stream_file)

    # Start thread to receive data
    thread = threading.Thread(target=receive_thread, args=(selector, ))
    thread.start()

    # Start measuring, this will start the stream of data from the module
    requests.post(base_url + "/measurements")
    print("Measurement started")

    # Print measurement status including any errors that may occur
    prev_tag, prev_status, start = 0, "", time.time()
    while time.time() - start < kwargs['time']:
        response = requests.get(base_url + "/onchange?last=" + str(prev_tag)).json()
        prev_tag = response["lastUpdateTag"]
        status = f'Measuring {response["recordingStatus"]["timeElapsed"]}'
        errors = get_measurement_errors(response["recordingStatus"]["channelStatus"])
        status = status + " " + ("OK" if len(errors) == 0 else  ", ".join(errors))
        if prev_status != status:
            print(status)
            prev_status = status

    # Stop measuring
    requests.put(base_url + "/measurements/stop")
    print("Measurement stopped")

    # Close the streaming connection, data file, and recorder
    stream_sock.close()
    thread.join()
    stream_file.close()

    requests.put(base_url + "/finish")
    requests.put(base_url + "/close")

    print(f"Stream saved as {kwargs['file']}")
