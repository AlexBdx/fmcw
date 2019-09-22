import ast
import numpy as np
import time
import collections
import copy
from scipy.signal import butter, filtfilt
import io
from queue import Queue, Empty
from threading import Thread
import csv
import os
import multiprocessing as mp
from fmcw import ftdi, adc


import matplotlib
matplotlib.use('Qt5Agg')  # Use another backend
matplotlib.rc('image', cmap='jet')
import matplotlib.pyplot as plt
plt.ion()


def butter_highpass(cutoff, fs, order=4):
    """
    User friendly wrapper for a highpass scipy.signal.butter
    :param cutoff: cutoff frequency
    :param fs: sampling frequency
    :param order: order of the Butterworth filter
    :return: scipy butter objects
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=4):
    """
    Filter data with a highpass scipy.signal.butter
    :param data: Data to filter
    :param cutoff: Cutoff frequency
    :param fs: Sampling frequency
    :param order: Order of the Butterworth filter
    :return: Filtered data
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def twos_comp(val, bits):
    """
    Compute the 2's complement of int value val
    :param val: Bytes to complement
    :param bits:
    :return: 2's complement of int value val
    """
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val                         # return positive value as is


def f_to_d(f, s):
    """
    Converts frequency bins to distance bins based on ADC settings
    :param f: Frequency bins
    :param s: Settings dictionary
    :return: Distance bins
    """
    return s['c']*f/(2*(s['bw']/s['t_sweep']))

def create_bases(s):
    """
    Create the x axis data for all sorts of plots. This will speed up the display of the plots by caching it and
    limiting the amount of data to be redrawn.
    :param s: Settings dictionary
    :return: time, frequency, distance, angle bins
    """
    wl = s['c'] / (s['f0'] + s['bw'] / 2)  # [m] Center wavelength
    t = np.linspace(0, s['t_sweep'], s['sweep_length'])  # [s] Time base
    f = np.linspace(0, s['if_amplifier_bandwidth'] / 2, s['sweep_length'] // 2 + 1)  # [Hz] Frequency base
    d = f_to_d(f, s)  # [m] Distance base

    angles = 180 / np.pi * np.arcsin(np.linspace(1, -1, s['angle_pad']) * wl / (2 * s['d_antenna']))  # [°] Degree base
    return t, f, d, angles

def r4_normalize(x, d, e=1.5):
    """
    Not sure what this does. Used when processing the angle data
    :param x:
    :param d:
    :param e:
    :return:
    """
    n = d[-1]**e
    return x*d**e/n


def read_settings(f, encoding=None):
    """
    Reads the first line of a file and evaluates it as python code. Used when reading the binary log as the first line
    contains the settings dictionary.
    :param f: File handle
    :param encoding:
    :return: Settings dictionary from string evaluated as python code
    """
    f.seek(0)
    data = f.readline()
    data = data.decode(encoding) if encoding else data
    settings = ast.literal_eval(data)
    return settings

def find_start_batch(data, s, initial_index=0):
    """
    Find the starting index of the first valid batch of sweep data and its corresponding header.
    :param data: Batch of data coming from the FPGA via the USB port
    :param s: Settings dictionary
    :param initial_index: 0 if reading a new batch, non zero if finding the next valid sweep within a batch
    :return: Starting index of a sweep data, header of that sweep
    """
    flag_valid_header = False
    for index in range(initial_index, len(data) - s['nbytes_sweep'] - 2):
        current_header = [data[index], data[index + 1]]
        if current_header[0] == s['start'] and data[index + s['nbytes_sweep'] + 2] == s[
            'start']:  # Not 100% foolproof, but cannot be anyway
            print("[INFO] Found start signal {} at position {} (jumped {} byte)."
                  .format(current_header, index, index-initial_index))
            print("[INFO] Next header would read [{}, {}]".format(data[index + s['nbytes_sweep'] + 2],
                                                                  data[index + s['nbytes_sweep'] + 3]))
            flag_valid_header = True
            break
    if flag_valid_header:  # All good
        index += 2  # Skip the header, it is saved in current_header
    else:
        # index = -1  # No valid header was found
        index = 0
        current_header = [0, 0]
        #print("[ERROR] index: {} | len(data): {} | s['nbytes_sweep']: {}".format(index, len(data), s['nbytes_sweep']))
        #raise ValueError('[ERROR] No valid header found in the data!')

    assert index >= 0
    assert current_header[0] == 127 or current_header[0] == 0
    return index, current_header

def process_batch(rest, data, s, next_header, counter_decimation, sweep_count, verbose=False):
    """
    Main function to process incoming batches of data from the FPGA. The goal is to find valid sweeps in the data. Main 
    challenges are that the start of the data might come from the end of a previous sweep, there might be some dropped 
    byte in some sweeps due to latency from the OS vs real time FPGA, and a last sweep that is incomplete and has to be 
    merged with the next batch.
    :param rest: End of the previous batch that was not long enough to constitute a whole sweep.
    :param data: New batch of USB data from the FPGA
    :param s: Settings dictionary
    :param next_header: Expected header of the next sweep
    :param counter_decimation: Rolling counter, keeps track of software decimation across batches
    :param sweep_count: Global number of valid, post decimation sweeps that have been found
    :param verbose: A lot of extra info will be displayed
    :return: batch_ch, next_header, rest, sweep_count, counter_decimation
    """
    # Sanity checks on types
    assert type(rest)==bytes or rest == None
    assert type(data)==bytes
    assert type(s['start'])==int
    assert type(s['nbytes_sweep'])==int
    assert type(next_header)==list

    # Sanity checks on length
    assert len(next_header) == 2
    assert type(next_header[0] == np.int8) and type(next_header[1] == np.int8)
    assert len(data) + len(rest) > 0

    # Sanity check on nature of next_header
    assert next_header[0] in [0, 1, s['start']]  # Can be the start signal or an error code

    if len(rest) > len(data):
        print(len(data), len(rest))
    # 0. Create temp variables
    #counter_decimation = 0  # For software decimation
    sweeps_scanned = 0
    skipped_data = np.zeros((s['nbytes_sweep'] // 2,), dtype=np.int16)  # Create the 0 array only once
    batch_ch = dict()
    for k in range(s['channel_count']):
        batch_ch[k + 1] = []
        # data_2[k + 1] = []
    batch_ch['skipped_sweeps'] = []  # Stores the skipped frames for both channels

    # 1. Concatenate the rest with the data
    initial_data_length = len(data)  # DEBUG
    data = rest + data  # Concatenate the bytearrays
    if len(data) > s['patience_data_length']:  # Threshold in data length
        raise ValueError('[ERROR] No valid header found in these {} byte of data'.format(len(data)))

    if verbose:
        print("[INFO] rest length: {} | block: {} long | shape to process: {}".format(len(rest), initial_data_length, len(data)))
    assert type(data) == bytes

    # 2. Find the start
    if next_header[0] != s['start']:  # The start was not found
        # This can happen if the previous sweep ended with a lot of data but no valid header.
        # All the remaining data was carried over as rest. Once a new valid header is found, we can see how many
        # sweeps were skipped and fill batch_ch with the corresponding amount of 0s. Should be rare to skip so many
        # sweep unless very few bytes are read on the USB port.

        # CASE 1: [0, 0] - No valid header ever found
        if next_header[0] == 0:  # This is the first time we go through process_batch
            index, next_header = find_start_batch(data, s)
            if next_header[0] == 0:  # valid header not found - need more data to get started
                print("[WARNING] No valid header found when searching for a start signal for a case [0, 0]")
                rest = data
                return batch_ch, next_header, rest, sweep_count, counter_decimation
            else:
                # Found a valid header. Let's get started with the main while loop!
                assert next_header[0] == s['start']

        # CASE 2: [1, N] - The backward search failed previously
        elif next_header[0] == 1:  # The previous batch ended without a valid next header but has a frame_number
            current_frame_number = next_header[1]
            index, next_header = find_start_batch(data, s)  # Start searching from the start
            if next_header[0] == 0:  # valid header not found - need more data to get started
                print("[WARNING] No valid header found when searching for a start signal for a case [1, N]")
                next_header = [1, current_frame_number]  # Restore header
                rest = data
                return batch_ch, next_header, rest, sweep_count, counter_decimation
            else:  # Sucess! A valid header was found. Append the required 0s for the skipped sweeps
                assert next_header[0] == s['start']
                print("[INFO] Recovered from backward search failure with", next_header)
                sweeps_scanned = (next_header[1] - current_frame_number)&0xff
                for _ in range(sweeps_scanned):  # Add all the required zeros before getting started
                    if counter_decimation == s['soft_decimate']:
                        signed_data = skipped_data  # Will append just 0s. Frame is skipped
                        batch_ch['skipped_sweeps'].append(sweep_count)

                        for channel in range(s['channel_count']):  # Channels number are 1 based for now
                            batch_ch[channel + 1].append(signed_data[channel::s['channel_count']])  # Data is entangled

                        # Increment counters
                        counter_decimation = 0
                        sweep_count += 1  # Only count sweeps after decimation
                    else:  # Decimate sweep: should we interpolate the data or drop zeros?
                        if verbose:
                            print("[INFO] Decimating sweep : {}/{}".format(counter_decimation, s['soft_decimate']))
                        counter_decimation += 1
                sweeps_scanned = 0
                print("[INFO] Done recovering ", next_header)

    else:  # There is a valid previous header!
        index = 0  # rest+data will be scanned from the start
        if verbose:
            print("[INFO] Starting with header", next_header)

    try:  # [DEBUG] I saw that fail once, not sure why
        current_frame_number = next_header[1]
    except:
        print(next_header)


    assert type(data) == bytes
    assert type(next_header[0]==np.int8)
    try:
        assert next_header[0] == s['start']  # All issues must have been solved before getting to the main loop
    except:
        print(next_header)
        raise
    assert index >= 0  # Cannot be negative otherwise creates a wreck

    # 3. Process the batches as long as they are valid.
    while index+s['nbytes_sweep']+2 < len(data) and next_header[0] == s['start']:
        # 3.1 Scoop the next s['nbytes_sweep'] and the following header
        if verbose:
            print("\n[INFO] Reading sweep", current_frame_number)
        batch = data[index:index+s['nbytes_sweep']]
        if len(batch) == 0:
            print("Debug 2")
            print(index, len(data), s['nbytes_sweep'])
            raise
        next_header = [data[index+s['nbytes_sweep']], data[index+s['nbytes_sweep']+1]]

        # 3.2 First case: the header is valid
        if next_header[0] == s['start'] and next_header[1] == (current_frame_number+1)&0xff:
            if verbose:
                print("[INFO] Successfully read sweep {} starting at index {}".format(current_frame_number, index))
            flag_success = True
            index += s['nbytes_sweep'] + 2
            sweeps_scanned = (next_header[1] - current_frame_number) & 0xff
            assert sweeps_scanned == 1  # Debug

        # Second case: the next_header does not match expectations
        else:  # Drop this sweep as next_header does not match expectations
            if verbose:
                print('[WARNING] Lost track of sweep starting at {} '.format(index))
                print('[WARNING] Next header at {} read: {} | Expected: ({}, {})'
                      .format(index+s['nbytes_sweep'], next_header, s['start'], (current_frame_number+1)&0xff))
            flag_success = False

            # Option 1: Look for the start of the correct next_header in the dropped data
            for jj in range(s['nbytes_sweep'])[::-1]:  # Go in reverse
                try:  # DEBUG
                    if jj == s['nbytes_sweep'] - 1:
                        # if sweep_data[jj] == s['start'] and next_frame_number == (current_frame_number+1)&0xff:  # I think this is wrong
                        if batch[jj] == s['start'] and next_header[0] == (current_frame_number + 1)&0xff:  # Check this
                            next_header = [batch[jj], next_header[0]]  # Only 1 int16 was lost
                            break

                    elif batch[jj] == s['start'] and batch[jj + 1] == (current_frame_number + 1)&0xff:
                        next_header = [batch[jj], batch[jj+1]]  # Cast batch values to int and put them in a list
                        break
                except:
                    print("DEBUG")
                    print(jj, len(batch))
                    print(next_header)
                    assert 0
            # Process the result of that backward search
            if next_header == [s['start'], (current_frame_number + 1) & 0xff]:  # Valid header!
                if verbose:
                    print("[WARNING] Found header {} at {}".format(next_header, index+jj))
                    print("[WARNING] Skipping sweep {} from {} to {}.".format(current_frame_number, index, index+s['nbytes_sweep']))
                    print("[WARNING] Restarting with sweep {} from position {}".format(next_header[1], index+jj+2))
                index += jj + 2 # Skip the next_header and get ready to scoop s['nbytes_sweep'] of data
                sweeps_scanned = (next_header[1] - current_frame_number) & 0xff
                assert sweeps_scanned == 1  # Debug
            else:  # Backward pass failed
                # Option 2: Search for the header forward
                print("[WARNING] Failed backward search. Header must have been lost.")
                next_header = [1, (current_frame_number+1)& 0xff]  # Signal that the backward search failed
                assert jj == 0
                index += s['nbytes_sweep']  # Skip the whole "expected" sweep.
                sweeps_scanned = 1  # This sweep will be ignored
                """[TBR] Previously tried to find forward the next header
                index, next_header = find_start_batch(data, s, initial_index=index)
                print("[WARNING] Current sweep number: {} | Next header count: {}"
                      .format(current_frame_number, next_header[1]))

                if next_header[0] == 0:  # Failed to find a valid header forward
                    index = len(data)  # No rest will be generated
                    next_header = [0, current_frame_number]  # Communicate
                    sweeps_scanned = 0  # Skip channel assignement
                    print('[WARNING] Next header not found in previous incorrect frame nor rest of frame.')
                else:  # Valid header found in forward data
                    sweeps_scanned = (next_header[1] - current_frame_number) & 0xff  # Could be > 1
                """

        current_frame_number = next_header[1]  # Ready to read the next sweep

        # 3.3 Append data if we are not decimating
        #if counter_decimation == s['soft_decimate']:
        for _ in range(sweeps_scanned):
            if counter_decimation == s['soft_decimate']:
                if flag_success:
                    signed_data = np.frombuffer(batch, dtype=np.int16)  # Read as int16
                else:
                    signed_data = skipped_data # Will append just 0s. Frame is skipped
                    batch_ch['skipped_sweeps'].append(sweep_count)
    
                for channel in range(s['channel_count']):  # Channels number are 1 based for now
                    batch_ch[channel + 1].append(signed_data[channel::s['channel_count']])  # Data is entangled
    
                # Increment counters
                counter_decimation = 0
                sweep_count += 1  # Only count sweeps after decimation
            else: # Decimate sweep: should we interpolate the data or drop zeros?
                if verbose:
                    print("[INFO] Decimating sweep : {}/{}".format(counter_decimation, s['soft_decimate']))
                counter_decimation += 1
        sweeps_scanned = 0


    # 4. Finalization
    # Return numpy arrays rather than lists
    for k in range(s['channel_count']):
        batch_ch[k + 1] = np.array(batch_ch[k + 1], dtype=np.int16)
    rest = data[index:]  # Get the rest if index < length(data) else []
    if verbose:
        print("\n[INFO] There is a rest of length", len(rest))
    
    return batch_ch, next_header, rest, sweep_count, counter_decimation


def calculate_if_data(sweeps, s):
    """
    Convert the raw data to a differential voltage level. Note that the data is cast from int16 to float64.
    :param sweeps: Sweeps to consider
    :param s: Settings dictionary
    :return: Voltage is returned as a dict with each key being a channel.
    """
    assert type(sweeps) == dict
    if_data = {}

    for channel in sweeps: # Go through all available channels
        data = np.array(sweeps[channel], dtype=np.float)
        data *= 1 / (s['fir_gain'] * 2 ** (s['adc_bits'] - 1))  # No w
        if_data[channel] = data

    return if_data


def calculate_angle_plot(sweeps, s, tfd_angles):
    """Perform the data processing to calculate the angular location of objects in a single sweep. The goal is to plot
    that result afterward, not to process multiple sweeps.

    :param sweeps: Data from which the angle position will be calculated
    :param s: Settings dictionary
    :param tfd_angles: Tuple containing all the bins important for the plotting
    :return: fxdb
    """
    # WARNING: ONLY 2 CHANNELS SUPPORTED SO FAR
    assert type(sweeps) == dict
    d = tfd_angles[2]
    angle_mask = tfd_angles[4]

    fxm = None  # If not None need to see sth else
    if fxm:
        coefs = [0.008161818583356717,
                 -0.34386493885120994,
                 0.65613506114879,
                 -0.34386493885120994,
                 0.008161818583356717]
    else:
        coefs = [1]
    angle_window = np.kaiser(s['angle_pad'], 150)

    a = np.fft.rfft(sweeps[1])  # Channels indexes are 1 based
    b = np.fft.rfft(sweeps[2])
    # b *= np.exp(-1j*2*np.pi*channel_dl/(s['c']/(s['f0']+s['bw']/2)))
    b *= np.exp(-1j * 2 * np.pi * s['channel_offset'] * np.pi / 180)

    if s['swap_chs']:
        x = np.concatenate((b, a)).reshape(2, -1)
    else:
        x = np.concatenate((a, b)).reshape(2, -1)

    fx = np.fft.fftshift(np.fft.fft(x, axis=0, n=s['angle_pad']), axes=0)
    fx = r4_normalize(fx, d)

    if 0:  # Calculate the min/max to use with the color bar
        max_range_i = np.searchsorted(d, s['max_range'])
        cblim = np.max(20 * np.log10(np.abs(fx[:max_range_i, :]))) + 10
        cblim = [cblim-50, cblim]  # min max array

    if fxm is None: # Apply coefficients ?
        fx = coefs[0] * fx
    else:  # k is not defined anymore, would have to get it from original files
        fx += coefs[k] * fx

    if s['flag_Hanning']:  # Apply a Hanning window to the peak
        result = []
        center = fx.shape[0] / 2
        for freq in np.transpose(fx):  # Transpose for convenience
            m = np.argmax(np.abs(freq))  # Find the index of the max
            window = np.roll(angle_window, int(round(-center - m)))  # Center window on max
            freq *= window  # Apply the window
            result.append(freq)
        fx = np.transpose(np.stack(result))  # Get back to original shape

    fx = fx[angle_mask]
    fxdb = 20 * np.log10(np.abs(fx))


    return fxdb


def calculate_range_time(ch, s, single_sweep=-1):
    """
    Take a single sweep and calculate the distances of all signals. All the channels are averaged in a single virtual
    channel. While this is not super good practice, it is mostly okay given how far the objects are in comparison to the
    distance between antennas.
    :param ch: dict containing the sweep data for each channel
    :param s: Settings dictionary
    :param single_sweep: Sweep to select in the dictionary in case there are actually multiple of them. To be removed.
    :return: im, nb_sweeps, max_range_index
    """
    # WARNING: ONLY USING CHANNEL 2 FOR THAT
    # Take the average of all channels
    sweeps = np.zeros(ch[s['active_channels'][0]].shape)  # There is at least one active channel
    for channel in s['active_channels']:
        sweeps += ch[channel].astype(np.int64)
    sweeps /=  s['channel_count']  # Becomes a single 2D numpy array


    if len(ch[s['active_channels'][0]].shape) == 1:  # Only 1 sweep was given
        nb_sweeps = 1
        single_sweep = True
    else:  # Extract a sweep from an ndarray of them
        if single_sweep != -1:
            sweeps = sweeps[single_sweep]  # Using last sweep only
            nb_sweeps = 1
            single_sweep = True
        else:
            nb_sweeps = sweeps.shape[0]  # Number of sweeps
            single_sweep = False
    sweep_length = s['sweep_length']

    """[TBR] Potentially subtract the background & all"""
    if s['subtract_background']:
        background = []
        for i in range(sweep_length):
            x = 0
            for j in range(len(sweeps)):
                x += sweeps[j][i]
            background.append(x / len(sweeps))

    max_range_index = int(sweep_length * s['max_range'] / s['range_adc'])
    max_range_index = min(max_range_index, sweep_length // 2)

    im = np.zeros((max_range_index - 2, nb_sweeps))
    w = [1]*sweep_length if single_sweep else np.kaiser(sweep_length, s['kaiser_beta'])

    for e in range(nb_sweeps):
        sw = sweeps if single_sweep else sweeps[e]
        if s['subtract_clutter'] and e > 0:
            sw = [sw[i] - sweeps[e - 1][i] for i in range(sweep_length)]
        if s['subtract_background']:
            sw = [sw[i] - background[i] for i in range(sweep_length)]

        sw = [sw[i] * w[i] for i in range(len(w))]  # Take a Kaiser window of the sweep
        fy = np.fft.rfft(sw)[3:max_range_index + 1]  # FFT of the sweep
        fy = 20 * np.log10((s['adc_ref'] / (2 ** (s['adc_bits'] - 1) * s['fir_gain'] * max_range_index)) * np.abs(fy))
        fy = np.clip(fy, -100, float('inf'))
        im[:, e] = np.array(fy)
        im = np.array(fy) if single_sweep else im

    if 0:
        cblim = [min(im), max(im, 0)]


    return im, nb_sweeps, max_range_index


def find_start(f, start, s):
    """
    Find a valid start header in a binary file by looking for two valid headers separated by the proper length of data.
    Given the simplicity of the system, it is not possible to guarantee that this data is "legit" as valid headers could
    be coming from random data. However, it is very unlikely.
    :param f: File handle
    :param start: Start signal to look for
    :param s: Settings dictionary
    :return: The current file.seek() index at which the valid data starts and the corresponding frame number. It is
    coded on a single byte, so expect it to roll over after 255 is reached.
    """
    done = False
    while not done:
        r = f.read(1)
        if r == '':  # Would indicate that we have reached the EOF
            return
        if r != start:
            continue

        else:  # Found start character
            current_frame_number = f.read(1)[0]
            done = True
            current_position = f.tell()
            # Verify that what follows are full sweeps
            for j in range(1):
                f.read(s['nbytes_sweep'])  # Read a whole sweep
                if f.read(1) != start:
                    done = False
                next_frame_number = f.read(1)

                if len(next_frame_number) == 0:
                    return
                if next_frame_number[0] != current_frame_number+1+j:
                    done = False
            f.seek(current_position)  # Go back to just before the data
    return current_position, current_frame_number


def import_data(f, start, first_frame, s, samples, verbose=False):
    """
    Import the data from a binary file. This was the source inspiration for process_batch, which is more up to date and
    deal with real time data. As a result, this might not be fully up to date.
    :param f: File handle
    :param start: Start signal for the headers
    :param first_frame: Get the current frame number read from find_start
    :param s: Settings dictionary
    :param samples: Legacy argument, useless
    :param verbose: Print a lot more info
    :return:
    """
    counter_decimation = 0
    signal = s['start']
    sweep_count = 0

    current_frame_number = first_frame
    # The data will be stored in a dict of lists, the keys being the channel number
    data = dict()
    for k in range(s['channel_count']):
        data[k+1] = []
    data['skipped_sweeps'] = []  # Stores the skipped frames for both channels

    while samples == None or i < samples:
        # Read the start signal and the the frame_number
        if verbose:
            print("\n[INFO] Current header [{}, {}] | sweep_data starting at: {}".format(signal, current_frame_number, f.tell()))
        t0 = time.perf_counter()
        sweep_data = f.read(s['nbytes_sweep'])  # Block read
        t1 = time.perf_counter()
        if len(sweep_data) != s['nbytes_sweep']:
            break  # No more data, we have reached the end of the file

        # Read the header
        signal, next_frame_number = f.read(2) # Should get the next signal and frame_number
        restart = False
        if signal != s['start']:
            if verbose:
                print('[WARNING] Lost track of start at {} | Next header read: [{}, {}] but expected: [{}, {}]'
                  .format(f.tell(), signal, next_frame_number, s['start'], (current_frame_number+1)&0xff))
            restart = True
        if restart == False and current_frame_number != None:
            if next_frame_number != (current_frame_number+1)&0xff:
                if verbose:
                    print('[WARNING] Lost a sweep at {} | Next header read: [{}, {}] but expected: [{}, {}]'.format(f.tell(), signal, next_frame_number, s['start'], (current_frame_number+1)&0xff))
                #assert 1==0
                restart = True

        if restart: # Find the nearest start flag, looking at the latest data first
            pos = f.tell()
            # Check in the data if a valid header can be found
            flag_success = False
            for jj in range(s['nbytes_sweep'])[::-1]:  # Go in reverse
                if jj == s['nbytes_sweep']-1:
                    if sweep_data[jj] == s['start'] and signal == (current_frame_number + 1) & 0xff:  # Check this if getting the chance
                        flag_success = True
                        break
                elif sweep_data[jj] == s['start'] and sweep_data[jj+1] == (current_frame_number+1)&0xff:
                    flag_success = True
                    break

            if flag_success:  # The next header was found in the previous sweep data: some data was dropped!
                if verbose:
                    print("[WARNING] Found next header [{}, {}] at position {} in the sweep_data of length {}".format(sweep_data[jj], sweep_data[jj+1], jj, s['nbytes_sweep']))
                # Sanity check:
                f.seek(pos - 2 - s['nbytes_sweep'] + jj)
                signal, next_frame_number = f.read(2)  # Drop previous sweep
                if verbose:
                    print('[WARNING] Jumped to {}, moved by {} byte'.format(f.tell(), f.tell()-pos))
                    print("[WARNING] Skipping sweep {}. New header: [{}, {}] (overall sweep count: {})".format(current_frame_number, signal, next_frame_number, sweep_count))
                # Process the new location
                current_frame_number = next_frame_number

                if f.tell()-pos > 0:
                    # Somehow the previous frame was s['nbytes_sweep'] and did not contain an issue
                    raise ValueError("[ERROR] Why was a correct header not found in the previous data?")

            else:
                raise ValueError('[ERROR] Next header not found in previous incorrect frame. Where is it?')
        else:
            current_frame_number = next_frame_number


        if counter_decimation == s['soft_decimate']:
            if verbose:
                print("[INFO] Using this sweep : {}/{}".format(counter_decimation, s['soft_decimate']))
            t0 = time.perf_counter()
            signed_data = np.frombuffer(sweep_data, dtype=np.int16)  # Does everything at once
            t1 = time.perf_counter()

            if restart:
                if verbose:
                    print("[WARNING] Due to restart, appending zeros for sweep {} (overall sweep counter: {})".format(current_frame_number-1, sweep_count))
                signed_data = np.zeros((s['nbytes_sweep']//2,), dtype=np.int16)
                data['skipped_sweeps'].append(sweep_count)
            # Append channel data to respective list
            for channel in range(s['channel_count']): # Channels number are 1 based for now
                data[channel+1].append(signed_data[channel::s['channel_count']])
            #print(data[1][0][:10])
            #print(data[2][0][:10])
            #assert not np.array_equal(data[1], data[2])
            sweep_count += 1
            counter_decimation = 0
        else: # Decimate sweep: should we interpolate the data or drop zeros?
            if verbose:
                print("[INFO] Decimating sweep : {}/{}".format(counter_decimation, s['soft_decimate']))
            counter_decimation += 1

    # Return numpy arrays rather than lists
    for k in range(s['channel_count']):
        data[k+1] = np.array(data[k+1], dtype=np.int16)
    return data


def compare_ndarrays(a, b):
    """Check if two arrays are equivalent or not with additional details
    Helper function written to find quickly why two arrays are not equal element wise.
    :param a: Array 1
    :param b: Array 2
    :return: Void. An exception is raised if a difference between the two arrays have been found.
    """
    print("Type: {} | {}".format(type(a), type(b)))
    if type(a) != type(b):
        raise TypeError("Arrays are not of the same type")
    
    print("Shape: {} | {}".format(a.shape, b.shape))
    if a.shape != b.shape:
        raise TypeError("Arrays do not have the same shape")
    
    print("Data type: {} | {}".format(a.dtype, b.dtype))
    if a.dtype != b.dtype:
        raise TypeError("Array elements do not have the same type")
    c = a-b
    c = c.ravel()
    index_c_non_zero = [i for i in range(len(c)) if c[i] != 0]
    print("Indexes where the difference is not 0 (count: {}):".format(len(index_c_non_zero)))
    print(index_c_non_zero)
    print(a[0])
    print(b[0])


def subtract_background(channel_data, w, data):
    """DEPRECATED?
    Subtract the mean to a list of sweeps and multiply the result by the weights w. One thing to note, is that sweeps
    full of zeros (coming from corrupted usb data) are left invariant.
    :param channel_data: dict of channels containing the sweep data as numpy arrays
    :param w: weights to apply to the array of sweeps
    :param data: Not sure
    :return: Updated channel_data
    """
    zero_sweep = np.zeros((channel_data.shape[1],), dtype=np.int16)
    # Subtract background from the channels data, which are numpy arrays
    # Input is the data dictionnary, output is the list of processed data
    background = np.sum(channel_data, axis=0)/(channel_data.shape[0]-len(data['skipped_sweeps']))
    channel_data = w*(channel_data - background)
    for skipped_frame in data['skipped_sweeps']:
        channel_data[skipped_frame] = zero_sweep
    return channel_data


def subtract_clutter(channel_data, w, data, clutter_averaging=1):
    """DEPRECATED?
    Subtract to a sweep the average of the previous clutter_averaging sweeps. It's some kind of moving average. The
    goal is to perform motion detection a lot more easily.
    :param channel_data: dict of channels containing the sweep data as numpy arrays
    :param w: weights to apply to the array of sweeps
    :param data: Not sure
    :param clutter_averaging: Number of previous sweeps to average before subtracting them to the current one.
    :return:
    """
    zero_sweep = np.zeros((channel_data.shape[1],), dtype=np.int16)
    a = np.zeros((clutter_averaging, channel_data.shape[1]))  # Padding for the first clutter_averaging sweeps
    print(a.shape, channel_data.shape)

    print(sorted(data['skipped_sweeps']))

    # Build the indexes to use for the clutter subtraction
    last_good_indexes = collections.deque(maxlen=clutter_averaging)  # Deque stored the last meaningful sweeps
    last_good_indexes.append(0)  # Assume the first sweep is okay
    subtract_channel_data = []
    for sweep_number in range(len(channel_data)):
        if sweep_number in sorted(data['skipped_sweeps']):
            subtract_channel_data.append(zero_sweep)
        else:
            accumulator_channel_data = zero_sweep.astype(np.float64)  # Temporarily switching to the float space

            # CUSTOM WEIGHTS FOR MOVING AVERAGE
            # WARNING: index 0 carries the oldest element in the deque, you probably want a lower weight on it.
            weights = [1 for index in range(len(last_good_indexes))]
            # weights = [1/(len(last_good_indexes)-index) for index in range(len(last_good_indexes))]
            if weights[0] > weights[-1]:
                print("[WARNING] Currently weighting the oldest sweeps more than the youngest. Are you sure?")
            # print(weights)
            assert len(weights) == len(last_good_indexes)
            for index, ii in enumerate(last_good_indexes):
                accumulator_channel_data += weights[index]*channel_data[ii]
            # if len(last_good_indexes):
                # print("[WARNING] Divide by zero at ", ii)
            accumulator_channel_data /= np.sum(weights)  # Divide by the sum of the weights used
            # Sanity check, verify that casting back to np.int16 is seamless
            assert np.max(accumulator_channel_data) < 2 ** 16
            assert np.min(accumulator_channel_data) > -2 ** 16
            subtract_channel_data.append(accumulator_channel_data)
            last_good_indexes.append(sweep_number)
    # Cast back to np.int16
    subtract_channel_data = np.array(subtract_channel_data, dtype=np.int16)
    print(subtract_channel_data.shape)
    channel_data = w*(channel_data - subtract_channel_data)
    #assert np.array_equal(channel_data_2[0], w*channel_data[0])  # Verify that the first item did not get subtracted anything
    return channel_data


class Writer(Thread):
    """
    Writer object that writes data to file. Created as a separate thread fed from a queue, so it's not blocking.
    Nothing special about it. Comes in twpo flavors:
    - Writer for binary files
    - Writer for csv files
    """
    def __init__(self, queue, s, encoding='latin1'):
        """Create the files (settings, binary log and csv log). Uniqueness is ensured by a timestamp prefix.

        :param queue: Input queue from which the data will be read. If the queue times out, the thread will terminate.
        :param s: Settings dictionary
        :param encoding: Depending on its value, a writer to binary file or csv file is created.
        """
        Thread.__init__(self)  # Mandatory call to the super constructor

        self.queue = queue  # input queue
        self.encoding = encoding  # type of writer
        self.wrote = 0  # Number of byte or lines written

        if self.encoding == 'latin1':
            self.f = open(s['path_raw_log'], 'w', encoding=self.encoding)
        elif self.encoding == 'csv':
            self.f = open(s['path_csv_log'], 'w')
            self.writer = csv.writer(self.f)
            self.writer.writerow(['Timestamp', 'Sweep number', 'Channel', 'Data'])  # Add a header
        elif self.encoding == 'settings':
            # Initialize the settings file and write the settings to file with csv.DictWriter
            with open(s['path_settings'], 'w') as f:  # Write the settings to file
                writer = csv.DictWriter(f, fieldnames=s.keys())
                writer.writeheader()
                writer.writerow(s)
            print("[INFO] Wrote the settings to {}".format(s['path_settings']))
        else:
            raise ValueError('[ERROR] File encoding method {} unknown'.format(self.encoding))
        self.timeout = s['timeout']


    def run(self):
        """
        Process the data from the queue and write it to file.
        :return:
        """
        while True:
            try:
                d = self.queue.get(True, self.timeout)
            except Empty:  # The queue has timed out. Not supposed to happen
                if self.encoding == 'csv':
                    print('[WARNING] {} Writer timed out after {} s without data | Wrote {} numbers to file'
                          .format(self.encoding, self.timeout, self.wrote))
                elif self.encoding == 'latin1':
                    print('[WARNING] {} Writer timed out after {} s without data | Wrote {} row to file'
                          .format(self.encoding, self.timeout, self.wrote))
                self.f.close()
                return

            if len(d) == 0:  # A '' signal was intentionally put to queue to indicate the end of the recording
                print('\n[INFO] Done after writing {:,} rows to file'.format(self.wrote))
                self.f.close()
                return
            else:
                if self.encoding == 'latin1':
                    self.f.write(d)
                elif self.encoding == 'csv':
                    self.writer.writerow(d)
                self.wrote += len(d)


def move_figure(f, number):
    """Move a figure to position (x, y) of the screen determined by the figure "number". Only 3 positions supported.
    DO NOT REALY ON THIS FUNCTION. CANNOT BE GENERALIZED TO OTHER USE CASES THAN WHAT IT WAS DESIGNED FOR.
    Basically, only used it with Qt5Agg. Did not try other backends and the code is not complete for it. They are slower
    than Qt when I tried, so not relevant. All units are px
    :param f: Figure handle
    :param number: Figure number. Only handles 3 different positions on screen, all 3 horizontal.
    :return: Void
    """
    max_width = 1900  # Ideally, should find the screen dimensions in a backend neutral way
    width = 600  # Width of the figure
    width_pad = 75  # Separation between the figures
    height = int(width*550/640)
    y = 1  # (1, 1) is upper left corner. The coordinates are 1 indexed, so (0, 0) will raise an exception.
    x = 1 + (number-1)*(width + width_pad)
    if not 1 <= number <= 3:
        raise ValueError('Only 3 plots can be displayed horizontally')

    backend = matplotlib.get_backend()  # Only works with Qt(5?)(Agg?) anyway
    # print("Backend is", backend)
    if backend == 'TkAgg':
        print(f.canvas.manager.window.maxsize())
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:  # Qt5Agg
        f.canvas.manager.window.setGeometry(x, y, width, height)
        print("The GUI's location is now", f.canvas.manager.window.geometry())


class if_display(mp.Process):
    """
    Sub-process for displaying the IF (Intermediate Frequency) data. These raw values coming out of the ADC (after
    FPGA filtering) are (almost) what make up the sweeps. There is a little bit of post-processing but not much.
    """
    def __init__(self, tfd_angles, s, data_accessible, new_sweep_if, sweep_to_display, time_stamp):
        """
        Nothing too special here. Store a bunch of information.
        :param tfd_angles: x axis data
        :param s: Settings dictionary
        :param data_accessible: shared mp.Event that acts as a Lock when the sweep data is being updated.
        :param new_sweep_if: shared mp.Event that signals when a new sweep is ready to be plotted for its IF.
        :param sweep_to_display: data to be displayed
        :param time_stamp: shared mp.Array that contains a few timestamp infos
        """
        mp.Process.__init__(self)  # Calling super constructor - mandatory
        # Create the window
        self.tfd_angles = tfd_angles
        self.s = s
        self.previous_sweep_counter = -self.s['refresh_stride'] # Virtual initial condition
        self.data_accessible = data_accessible
        self.new_sweep_if = new_sweep_if
        self.sweep_to_display = sweep_to_display
        self.time_stamp = time_stamp
        self.timing = []

    def run(self):
        """IF process loop

        :return:
        """
        # Create the figure object. Does not work when done in __init__ for some reason
        self.window = if_time_domain_animation(self.tfd_angles, self.s, grid=True, blit=True)

        # Main loop. Exits when the parent process terminates the process
        while True:
            # 1. Wait for the data flag to be set
            self.data_accessible.wait()
            self.new_sweep_if.wait()

            # 2. Retrieve the latest data and make a copy of it
            t0 = time.perf_counter()
            arr = np.array(self.sweep_to_display, dtype=np.int16)  # Makes a copy of that shared memory
            time_stamp = np.array(self.time_stamp)  # Same here, make a copy of the shared memory

            # 3. Reshape the sweep data as it had to be fit in a static, 1D, C-style array.
            arr = arr.reshape((self.s['channel_count'],
                               int(self.s['t_sweep'] * self.s['if_amplifier_bandwidth'])))
            data = {key:arr[index] for index, key in enumerate(self.s['active_channels'])}  # dict for compatibility

            # 4. Process the IF data
            if_data = calculate_if_data(data, self.s)

            # 5. Sanity checks: did we skip some refreshes for being too slow?
            sweeps_skipped = int(round((time_stamp[0] - self.previous_sweep_counter) / self.s['refresh_stride'])) - 1
            verbose = False
            if sweeps_skipped > 0 and verbose:
                # Some refreshes were skipped - not good!!
                print("[WARNING] Refresh rate cannot be sustained for IF data | [{}; {}[ (total: {}) were skipped"
                      .format(self.previous_sweep_counter,
                              self.previous_sweep_counter+sweeps_skipped*self.s['refresh_stride'],
                              sweeps_skipped))

            # 6. Update the figure
            self.window.update_plot(if_data, time_stamp)

            # 7. Set the necessary counters and flags
            self.previous_sweep_counter = int(time_stamp[0])
            self.new_sweep_if.clear()
            self.timing.append(time.perf_counter() - t0)
            #print("IF loop duration: mean: {:.3f} s | std: {:.3f} s".format(np.mean(self.timing), np.std(self.timing)))

    def __del__(self):
        print("[INFO] IF sub-process is now terminating.")


class if_time_domain_animation():
    def __init__(self, tfd_angles, s, grid=False, blit=False):
        """
        Initialize an object that will contain the IF plot.
        :param tfd_angles:
        :param s: Settings dictionary
        :param grid: Display the grid on the plot screen.
        :param blit: Blit is used to speed up image display by caching what is not redrawn.
        """
        # 1. Save the figure in this object
        t = tfd_angles[0]
        self.fig = plt.figure("IF time domain")
        move_figure(self.fig, 1)  # Figure 1 is in the top left corner

        # 2. Set up the static parts of the figure
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time [s]')
        self.ax.set_ylabel('Voltage [V]')
        self.lines = {}
        self.ax.set_xlim([0, t[-1]])
        if s['cblim_if'] != []:  # Set the color bar limits if they have been defined. Otherwise, dynamic.
            self.ax.set_ylim(s['cblim_if'])
        self.ax.grid(grid)

        # 3. Display initial data (zeros) to activate the figure
        self.fig.canvas.draw()  # note that the first draw comes before setting data
        for channel in s['active_channels']:  # Channels are 1 based due to hardware considerations
            self.lines[channel] = self.ax.plot(t, np.zeros((len(t),)), label='CH'+str(channel))[0]  # Grab first in list
        self.ax.legend(loc='best')
        plt.show(block=False)  # Display the figure

        # 4. If blitting, cache background
        self.blit = blit
        if self.blit:
            # cache the background
            self.axbackground = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def update_plot(self, if_data, time_stamp):
        """Dynamic refresh of the IF plot.
        A lot of work has been put in reducing the time necessary to refresh a plot. There must be some possible
        improvements, especially by messing with the backend directly.
        TO DO: less data points could be plotted as an entire sweep is likely to contain more points than pixels
        available to it on the screen.
        :param if_data: processed IF data from a sweep
        :param time_stamp: Timestamp for the current sweep
        :return:
        """
        for channel in if_data:  # Update y-data only, for all active channels
            self.lines[channel].set_ydata(if_data[channel])

        self.ax.set_title('IF time-domain at time T = {:.3f} s | FPGA time: {:.1f} s (lag: {:.1f} s)'
                          .format(time_stamp[1], time_stamp[2], time_stamp[2]-time_stamp[1]))  # Update title

        if self.blit:  # Given the performance boost, should be the default case
            self.fig.canvas.restore_region(self.axbackground)  # restore background
            for channel in if_data:  # redraw just the points
                self.ax.draw_artist(self.lines[channel])
            self.fig.canvas.blit(self.ax.bbox)  # fill in the axes rectangle

            self.fig.canvas.update()
            self.fig.canvas.flush_events()

        else:
            self.fig.canvas.flush_events()  # The order does not seem to matter with canvas.draw()
            self.fig.canvas.draw()

    def __del__(self):
        plt.close(self.fig.number)


class angle_display(mp.Process):
    """
    Sub-process to display the angular information coming from both receivers.
    """
    def __init__(self, tfd_angles, s, data_accessible, new_sweep_angle, sweep_to_display, time_stamp):
        """
        Nothing too special here. Store a bunch of information.
        :param tfd_angles: x axis data
        :param s: Settings dictionary
        :param data_accessible: shared mp.Event that acts as a Lock when the sweep data is being updated.
        :param new_sweep_if: shared mp.Event that signals when a new sweep is ready to be plotted for its Angle.
        :param sweep_to_display: data to be displayed
        :param time_stamp: shared mp.Array that contains a few timestamp infos
        """
        mp.Process.__init__(self)  # Mandatory call to super
        # Create the window
        self.s = s
        self.tfd_angles = tfd_angles
        self.previous_sweep_counter = -self.s['refresh_stride'] # Virtual initial condition
        self.data_accessible = data_accessible
        self.new_sweep_angle = new_sweep_angle
        self.sweep_to_display = sweep_to_display
        self.time_stamp = time_stamp
        self.timing = []

    def run(self):
        """
        Angle sub-process loop
        :return:
        """
        # Create the figure object. Does not work when done in __init__ for some reason
        self.window = angle_animation(self.tfd_angles, self.s, method='cross-range', blit=True)
        # Main loop. Exits when the parent process terminates the process
        while True:
            # 1. Wait for the data flag to be set
            self.data_accessible.wait()
            self.new_sweep_angle.wait()

            # 2. Retrieve the latest data and make a copy of it
            t0 = time.perf_counter()
            arr = np.array(self.sweep_to_display, dtype=np.int16)  # Makes a copy of that shared memory
            time_stamp = np.array(self.time_stamp)  # Same here, make a copy of the shared memory

            # 3. Reshape the sweep data as it had to be fit in a static, 1D, C-style array.
            arr = arr.reshape((self.s['channel_count'],
                               int(self.s['t_sweep'] * self.s['if_amplifier_bandwidth'])))
            data = {key: arr[index] for index, key in enumerate(self.s['active_channels'])}  # dict for compatibility

            # 4. Process the data to determine the angular components
            fxdb = calculate_angle_plot(data, self.s, self.tfd_angles)

            # 5. Sanity checks: did we skip some refreshes for being too slow?
            sweeps_skipped = int(round((time_stamp[0] - self.previous_sweep_counter) / self.s['refresh_stride'])) - 1
            verbose = False
            if sweeps_skipped > 0 and verbose:
                # Some refreshes were skipped - not good!!
                print("[WARNING] Refresh rate cannot be sustained for angle data | [{}; {}[ (total: {}) were skipped"
                      .format(self.previous_sweep_counter,
                              self.previous_sweep_counter + sweeps_skipped * self.s['refresh_stride'],
                              sweeps_skipped))

            # 6. Update the figure
            self.window.update_plot(fxdb, time_stamp)

            # 7. Set the necessary counters and flags
            self.previous_sweep_counter = int(time_stamp[0])
            self.new_sweep_angle.clear()
            self.timing.append(time.perf_counter() - t0)
            #print("Angle loop duration: mean: {:.3f} s | std: {:.3f} s".format(np.mean(self.timing), np.std(self.timing)))

    def __del__(self):
        print("[INFO] Angle sub-process is now terminating.")


class angle_animation():
    def __init__(self, tfd_angles, s, method='angle', blit=False):
        """
        Initialize an object that will contain the Angle plot
        :param tfd_angles:
        :param s: Settings dictionary
        :param method: Various types of angular plots are available
        :param blit: Blit is used to speed up image display by caching what is not redrawn.
        """
        # 1. Save the figure in this object
        d = tfd_angles[2]
        angles = tfd_angles[3]
        angles_masked = angles[tfd_angles[4]]

        self.fig = plt.figure("Angle")
        move_figure(self.fig, 2)  # Figure 2 is in the middle
        self.method = method
        fxdb = np.zeros((len(angles_masked), len(d)))  # For figure initialization only

        # 2. Select the type of figure to display
        if self.method == 'polar':  # Polar plot, looks cool but takes a lot of space (displays all 360 degrees)
            self.ax = self.fig.add_subplot(111, polar=True)
            self.ax.set_xlabel("???")
            self.ax.set_ylabel("???")
        elif self.method == 'cross-range':  # Best type, shows a cone of detection
            self.ax = self.fig.add_subplot(111)
            # self.ax.set_xlim([0, max_range])
            self.ax.set_xlabel("Range [m]")
            ylim = 90 * np.sin(angles_masked[0] * np.pi / 180)
            ylim = [-ylim, ylim]
            self.ax.set_ylabel("Cross-range [m]")
            r, theta = np.meshgrid(d, angles_masked * np.pi / 180)
            x = r * np.cos(theta)
            y = -r * np.sin(theta)
            self.ax.axis('equal')
        elif self.method == 'angle':
            self.ax = self.fig.add_subplot(111)
            self.quad = self.ax.pcolormesh(d, angles_masked, fxdb)
            self.ax.set_xlim([d[0], s['max_range']])
            self.ax.set_xlabel("Range [m]")
            self.ax.set_ylim([angles_masked[0], angles_masked[-1]])
            self.ax.set_ylabel("Angle [$^o$]")
        else:
            raise ValueError('[ERROR] Incorrect method for the angle plots')


        # 3. Display initial data (zeros) to activate the figure
        self.fig.canvas.draw()  # Get ready to cache this

        # 4. If blitting, cache background
        self.blit = blit
        if self.blit:
            # cache the background
            self.axbackground = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        # 5. Depending on type, display initial plot
        if self.method == 'polar':
            self.quad = self.ax.pcolormesh(angles_masked * np.pi / 180, d, fxdb.transpose())
        elif self.method == 'cross-range':
            self.quad = self.ax.pcolormesh(x, y, fxdb)
        elif self.method == 'angle':
            self.quad = self.ax.pcolormesh(d, angles_masked, fxdb)
        else:
            raise ValueError('[ERROR] Incorrect method for the angle plots')

        self.colorbar = self.fig.colorbar(self.quad, ax=self.ax)
        if s['cblim_angle'] != []:  # Set the color bar limits if they have been defined. Otherwise, dynamic.
            self.quad.set_clim(*s['cblim_angle'])

    def update_plot(self, fxdb, time_stamp):
        """Dynamic refresh of the angular plot.
        A lot of work has been put in reducing the time necessary to refresh a plot. There must be some possible
        improvements, especially by messing with the backend directly.
        TO DO: less data points could be plotted as an entire sweep is likely to contain more points than pixels

        :param fxdb: Angular data to plot
        :param time_stamp: Timestamp for the current sweep
        :return:
        """

        if self.method == 'polar':
            fxdb = fxdb.transpose()
        # https://stackoverflow.com/questions/18797175/animation-with-pcolormesh-routine-in-matplotlib-how-do-i-initialize-the-data
        # https://stackoverflow.com/questions/29009743/using-set-array-with-pyplot-pcolormesh-ruins-figure
        self.quad.set_array(fxdb[:-1, :-1].ravel())
        # self.quad.set_clim(*cblim)  # Updates the colorbar

        self.ax.set_title('Angle plot at time T = {:.3f} s | FPGA time: {:.1f} s (lag: {:.1f} s)'
                          .format(time_stamp[1], time_stamp[2], time_stamp[2] - time_stamp[1]))


        if self.blit:
            self.fig.canvas.restore_region(self.axbackground)  # restore background
            self.ax.draw_artist(self.quad)  # redraw just the points
            self.fig.canvas.blit(self.ax.bbox)  # fill in the axes rectangle

            self.fig.canvas.update()
            self.fig.canvas.flush_events()

        else:
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()


    def __del__(self):
        plt.close(self.fig.number)


class range_time_display(mp.Process):
    def __init__(self, tfd_angles, s, data_accessible, new_sweep_range_time, sweep_to_display, time_stamp):
        """
        Nothing too special here. Store a bunch of information.
        :param tfd_angles: x axis data
        :param s: Settings dictionary
        :param data_accessible: shared mp.Event that acts as a Lock when the sweep data is being updated.
        :param new_sweep_if: shared mp.Event that signals when a new sweep is ready to be plotted for its range time.
        :param sweep_to_display: data to be displayed
        :param time_stamp: shared mp.Array that contains a few timestamp infos
        """
        mp.Process.__init__(self)  # Mandatory call to super

        # Create the window
        max_range_index = int(s['sweep_length'] * s['max_range'] / s['range_adc'])
        self.max_range_index = min(max_range_index, s['sweep_length'] // 2)

        self.s = s
        self.previous_sweep_counter = -self.s['refresh_stride'] # Virtual initial condition
        self.data_accessible = data_accessible
        self.new_sweep_range_time = new_sweep_range_time
        self.sweep_to_display = sweep_to_display
        self.time_stamp = time_stamp
        self.timing = []

    def run(self):
        """
        Range time sub-process loop
        :return:
        """
        # Create the figure object. Does not work when done in __init__ for some reason
        self.window = range_time_animation(self.s, self.max_range_index, blit=True)
        # Main loop. Exits when the parent process terminates the process
        while True:
            # 1. Wait for the data flag to be set
            self.data_accessible.wait()
            self.new_sweep_range_time.wait()

            # 2. Retrieve the latest data and make a copy of it
            t0 = time.perf_counter()
            arr = np.array(self.sweep_to_display, dtype=np.int16)  # Makes a copy of that shared memory
            time_stamp = np.array(self.time_stamp)  # Same here, make a copy of the shared memory

            # 3. Reshape the sweep data as it had to be fit in a static, 1D, C-style array.
            arr = arr.reshape((self.s['channel_count'],
                               int(self.s['t_sweep'] * self.s['if_amplifier_bandwidth'])))
            data = {key: arr[index] for index, key in enumerate(self.s['active_channels'])}

            # 4. Process the data to determine the angular components
            im, nb_sweeps, max_range_index = calculate_range_time(data, self.s, single_sweep=0)

            # 5. Sanity checks: did we skip some refreshes for being too slow?
            sweeps_skipped = int(round((time_stamp[0]-self.previous_sweep_counter)/self.s['refresh_stride']))-1
            assert sweeps_skipped >= 0
            assert type(sweeps_skipped) == int
            verbose = False
            if sweeps_skipped > 0 and verbose:
                # Some refreshes were skipped - not good!!
                print("[WARNING] Refresh rate cannot be sustained for RT data | [{}; {}[ (total: {}) were skipped"
                      .format(self.previous_sweep_counter,
                              self.previous_sweep_counter + sweeps_skipped * self.s['refresh_stride'],
                              sweeps_skipped))

            # 6. Update the figure
            self.window.update_plot(im, time_stamp, sweeps_skipped)

            # 7. Set the necessary counters and flags
            self.previous_sweep_counter = int(time_stamp[0])
            self.new_sweep_range_time.clear()
            self.timing.append(time.perf_counter() - t0)
            #print("RT loop duration: mean: {:.3f} s | std: {:.3f} s".format(np.mean(self.timing), np.std(self.timing)))

    def __del__(self):
        print("[INFO] Range time is now terminating")


class range_time_animation():
    def __init__(self, s, max_range_index, blit=False):
        """
        Initialize an object that will contain the Range Time plot
        :param tfd_angles:
        :param s: Settings dictionary
        :param max_range_index: Maximum index used to display the requested range (or max dictated but Fourier)
        :param blit: Blit is used to speed up image display by caching what is not redrawn.
        """
        # 1. Save the figure in this object
        self.fig = plt.figure("Range-time")
        move_figure(self.fig, 3)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim([-s['real_time_recall'], 0])
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Range [m]")

        # 2. Set up the static parts of the figure
        nb_sweeps = int(s['real_time_recall']/s['refresh_period'])+1
        t = np.linspace(-s['real_time_recall'], 0, nb_sweeps, endpoint=True)
        x, y = np.meshgrid(t, np.linspace(0, s['range_adc']*max_range_index/s['sweep_length'], max_range_index-2))
        self.current_array = np.zeros((max_range_index-2, nb_sweeps))  # Store the value currently displayed for speed

        # 3. Cache the dynamic parts of the figure
        self.fig.canvas.draw()  # note that the first draw comes before setting data
        self.quad = self.ax.pcolormesh(x, y, self.current_array)
        self.colorbar = self.fig.colorbar(self.quad, ax=self.ax)


        # 4. If blitting, cache background
        self.blit = blit
        if self.blit:
            # cache the background
            self.axbackground = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        #self.quad.set_array(self.current_array[:-1, :-1].ravel())
        if s['cblim_range_time'] != []:  # Set the color bar limits if they have been defined. Otherwise, dynamic.
            self.quad.set_clim(*s['cblim_range_time'])

    def update_plot(self, im, time_stamp, sweeps_skipped):
        """Dynamic refresh of the Range time plot
        A lot of work has been put in reducing the time necessary to refresh a plot. There must be some possible
        improvements, especially by messing with the backend directly.
        TO DO: less data points could be plotted as an entire sweep is likely to contain more points than pixels
        available to it on the screen.
        :param im: range time data
        :param time_stamp: Timestamp for the current sweep
        :param sweeps_skipped: Important here to duplicate the current sweep as many times as sweeps we skipped
        :return:
        """
        # Roll current data along its columns as many times as sweeps we have skipped + 1
        self.current_array = np.roll(self.current_array, -sweeps_skipped-1, axis=1)

        # Replaced the rolled data with the new sweep data, duplicated as required
        im = im.reshape(-1, 1)  # Make the data 2D for broadcasting
        im = np.broadcast_to(im, (im.shape[0], sweeps_skipped+1))
        self.current_array[:, -sweeps_skipped-1:] =  im # Substitute the new array & pad

        self.quad.set_array(self.current_array[:-1, :-1].ravel())  # Flatten the data
        # self.quad.set_clim(*cblim)  # Updates the colorbar
        self.ax.set_title('Range time plot at time T = {:.3f} s | FPGA time: {:.1f} s (lag: {:.1f} s)'
                          .format(time_stamp[1], time_stamp[2], time_stamp[2]-time_stamp[1]))

        if self.blit:
            self.fig.canvas.restore_region(self.axbackground)  # restore background
            self.ax.draw_artist(self.quad)  # redraw just the points
            self.fig.canvas.blit(self.ax.bbox)  # fill in the axes rectangle

            self.fig.canvas.update()
            self.fig.canvas.flush_events()

        else:
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()

    def __del__(self):
        plt.close(self.fig.number)