#!/usr/bin/env python3
# pylint: disable=C0103

"""Example code to demonstrate streamed-data processing with LAN-XI Open API."""

import os
from datetime import datetime, timezone
import numpy as np
from kaitai.python.openapi_message import OpenapiMessage
import h5py

def calc_time(t):
    """
    Convert an Open API 'Time' structure to a number.
    Note Kaitai doesn't support the '**' operator, or we could have implemented
    a conversion function directly in the .ksy file.
    Args:
        t: an Open API 'Time' instance
    Returns:
        the time as a built-in, numeric type
    """
    family = 2**t.time_family.k * 3**t.time_family.l * 5**t.time_family.m * 7**t.time_family.n
    return t.time_count * (1 / family)

def get_quality_strings(l):
    """Given an 'l' list of validity objects, return a collection of descriptive strings."""
    strings = []
    for v in l:
        qs, prefix = "", ""
        if v["flags"].invalid:
            qs = qs + prefix + "Invalid Data"
            prefix = ", "
        if v["flags"].overload:
            qs = qs + prefix + "Overload"
            prefix = ", "
        if v["flags"].overrun:
            qs = qs + prefix + "Gap In Data"
            prefix = ", "
        if qs == "":
            qs = "OK"
        qs = f'{v["time"]}: ' + qs
        strings.append(qs)
    return strings

# parser = argparse.ArgumentParser()
# parser.add_argument("file", help="File containing Open API streaming data")
# parser.add_argument("-s", "--scale", action="store_true", \
#     help="Include this flag to scale the measurements by a different scale factor (inputRange x headroom / sensitivity) than what is stored in the TEDS data. If a json file with the microphone sesitivity is avaliable it will use the sensitivity value from there.")
# args = parser.parse_args()

def streaming_interpretation(**kwargs):
    print(f"Reading streaming data from file {kwargs['file']}...")
    file = os.path.join(kwargs['save_dir'],kwargs['file'])
    file_size = os.path.getsize(file)
    file_stream = open(file, 'rb')

    # Processed data will be stored in this collection
    data = {}

    while True:
        # Read the next Open API message from the file
        try:
            msg = OpenapiMessage.from_io(file_stream)
        except EOFError:
            print("")
            break

        # If 'interpretation' message, then extract metadata describing how to interpret signal data
        if msg.header.message_type == OpenapiMessage.Header.EMessageType.e_interpretation:
            for i in msg.message.interpretations:
                if i.signal_id not in data:
                    data[i.signal_id] = {}
                data[i.signal_id][i.descriptor_type] = i.value

        # If 'signal data' message, then copy sample data to in-memory array
        elif msg.header.message_type == OpenapiMessage.Header.EMessageType.e_signal_data:
            for s in msg.message.signals:
                if "start_time" not in data[s.signal_id]:
                    start_time = datetime.fromtimestamp(calc_time(msg.header.time), timezone.utc)
                    data[s.signal_id]["start_time"] = start_time
                if "samples" not in data[s.signal_id]:
                    data[s.signal_id]["samples"] = np.array([])
                more_samples = np.array(list(map(lambda x: x.calc_value, s.values)))
                data[s.signal_id]["samples"] = np.append(data[s.signal_id]["samples"], more_samples)

        # If 'quality data' message, then record information on data quality issues
        elif msg.header.message_type == OpenapiMessage.Header.EMessageType.e_data_quality:
            for q in msg.message.qualities:
                if "validity" not in data[q.signal_id]:
                    data[q.signal_id]["validity"] = []
                dt = datetime.fromtimestamp(calc_time(msg.header.time), timezone.utc)
                data[q.signal_id]["validity"].append({"time": dt, "flags": q.validity_flags})

        # Print progress
        print(f'{int(100 * file_stream.tell() / file_size)}%', end="\r")
        
    data_out = {}
    for index, (key, value) in enumerate(data.items()):
        # Scale samples using the scale factor from the interpretation message
        if kwargs['sensitivity']:
            scale_factor = 10*10**(1.5/20)/kwargs['sensitivity'][index]
        elif kwargs['calibrate']:
            scale_factor = 10*10**(1.5/20)
        else:
            scale_factor = value[OpenapiMessage.Interpretation.EDescriptorType.scale_factor]
        scaled_samples = (value["samples"] * scale_factor) / 2**23

        sample_rate = 1 / calc_time(value[OpenapiMessage.Interpretation.EDescriptorType.period_time])
        data_out.update({f'channel{index}':{'scale_factor':scale_factor,'sample_rate':sample_rate,'scaled_samples':scaled_samples}})


    with h5py.File(os.path.join(kwargs['save_dir'],kwargs['name']+'.h5'), 'a') as f:
        for k, v in data_out.items():
            if isinstance(v,dict):
                sub_group = f.create_group(k)
                for k2, v2 in v.items():
                    sub_group.create_dataset(k2, shape=np.shape(v2), data=v2)
            else:
                f.create_dataset(k, shape=np.shape(v), data=v)
