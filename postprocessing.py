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

import fmcw.display as display  # Cross imports are guettho

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=4):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def twos_comp(val, bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val                         # return positive value as is


def f_to_d(f, bw, sweep_duration):
    # Converts frequency bins to range bins
    c = 299792458.0  # [m/s] speed of light
    return c*f/(2*(bw/sweep_duration))

def create_bases(s):
    # Create the xlabel for all sorts of plots
    wl = s['c'] / (s['f0'] + s['bw'] / 2)  # [m] Center wavelength
    t = np.linspace(0, s['t_sweep'], s['SWEEP_LENGTH'])  # [s] Time base
    f = np.linspace(0, s['if_amplifier_bandwidth'] / 2, s['SWEEP_LENGTH'] // 2 + 1)  # [Hz] Frequency base
    d = f_to_d(f, s['bw'], s['t_sweep'])  # [m] Distance base
    angles = 180 / np.pi * np.arcsin(np.linspace(1, -1, s['angle_pad']) * wl / (2 * s['d_antenna']))  # [°] Degree base
    return t, f, d, angles

def r4_normalize(x, d, e=1.5):
    n = d[-1]**e
    return x*d**e/n


def read_settings(f, encoding=None):
    f.seek(0)
    data = f.readline()
    data = data.decode(encoding) if encoding else data
    settings = ast.literal_eval(data)
    return settings

def find_start_batch(data, s, initial_index=0):
    flag_valid_header = False
    for index in range(initial_index, len(data) - s['NBYTES_SWEEP'] - 2):  # index starts at 0
        next_header = [data[index], data[index + 1]]
        if next_header[0] == s['start'] and data[index + s['NBYTES_SWEEP'] + 2] == s[
            'start']:  # Not 100% foolproof, but cannot be anyway
            print("[INFO] Found start signal {} at position {} (jumped {} byte)."
                  .format(next_header, index, index-initial_index))
            print("[INFO] Next header would read [{}, {}]".format(data[index + s['NBYTES_SWEEP'] + 2],
                                                                  data[index + s['NBYTES_SWEEP'] + 3]))
            flag_valid_header = True
            break
    if flag_valid_header:  # All good
        index += 2  # Skip the header, it is saved in next_header
    else:
        index = -1  # No valid header was found
        next_header = []
        print("[WARNING] No valid header found when searching for a start signal")
        #print("[ERROR] index: {} | len(data): {} | s['NBYTES_SWEEP']: {}".format(index, len(data), s['NBYTES_SWEEP']))
        #raise ValueError('[ERROR] No valid header found in the data!')

    return index, next_header

def process_batch(rest, data, s, next_header, sweep_count, global_sweep_counter, verbose=False):
    # rest is a np.int8 array
    # data is a str
    # start is a binary str
    # s['NBYTES_SWEEP'] is an int
    # global_sweep_counter is an int64
    # Sanity checks
    assert type(rest)==bytes or rest == None
    assert type(data)==bytes
    assert type(s['start'])==int
    assert type(s['NBYTES_SWEEP'])==int
    assert type(next_header)==list
    # 0. Create temp variables
    #sweep_count = 0  # For software decimation
    sweeps_scanned = 0
    skipped_data = np.zeros((s['NBYTES_SWEEP'] // 2,), dtype=np.int16)  # Create the 0 array only once
    batch_ch = dict()
    for k in range(s['channel_count']):
        batch_ch[k + 1] = []
        # data_2[k + 1] = []
    batch_ch['skipped_sweeps'] = []  # Stores the skipped frames for both channels

    # 1. Concatenate the rest with the data
    initial_data_length = len(data)  # DEBUG
    if rest != None:
        data = rest + data  # Concatenate the bytearrays

    if verbose:
        print("[INFO] rest length: {} | block: {} long | shape to process: {}".format(len(rest), initial_data_length, len(data)))
    assert type(data) == bytes

    # 2. Find the start
    if next_header[0] == 0:  # Either no previous data or the previous frame did not end w/ valid header
        if next_header[1]:  # The previous batch ended without a valid next header
            current_frame_number = next_header[1]
            index, next_header = find_start_batch(data, s)
            sweeps_scanned = (next_header[1] - current_frame_number)&0xff

            for _ in range(sweeps_scanned):  # Add all the required zeros before getting started
                if sweep_count == s['sweeps_to_drop']:
                    signed_data = skipped_data  # Will append just 0s. Frame is skipped
                    batch_ch['skipped_sweeps'].append(global_sweep_counter)

                    for channel in range(s['channel_count']):  # Channels number are 1 based for now
                        batch_ch[channel + 1].append(signed_data[channel::s['channel_count']])  # Data is entangled

                    # Increment counters
                    sweep_count = 0
                    global_sweep_counter += 1  # Only count sweeps after decimation
                else:  # Decimate sweep: should we interpolate the data or drop zeros?
                    if verbose:
                        print("[INFO] Decimating sweep : {}/{}".format(sweep_count, s['sweeps_to_drop']))
                    sweep_count += 1
            sweeps_scanned = 0

        else:
            index, next_header = find_start_batch(data, s)
            assert len(next_header)  # If false, means that no valid start found in the whole batch!
    else:  # There is a valid previous header.
        index = 0  # rest+data will be scanned from the start
        if verbose:
            print("[INFO] Starting with header", next_header)

    try:  # [DEBUG] I saw that fail once, not sure why
        current_frame_number = next_header[1]
    except:
        print(next_header)


    assert type(data) == bytes
    assert type(next_header[0]==np.int8)
    # 3. Process the batches
    while index+s['NBYTES_SWEEP']+2 < len(data):  # As long as we can scoop the next batch and header
        # 3.1 Scoop the next s['NBYTES_SWEEP'] and the following header
        if verbose:
            print("\n[INFO] Reading sweep", current_frame_number)
        batch = data[index:index+s['NBYTES_SWEEP']]
        next_header = [data[index+s['NBYTES_SWEEP']], data[index+s['NBYTES_SWEEP']+1]]

        # 3.2 First case: the header is valid
        if next_header[0] == s['start'] and next_header[1] == (current_frame_number+1)&0xff:
            if verbose:
                print("[INFO] Successfully read sweep {} starting at index {}".format(current_frame_number, index))
            flag_success = True
            index += s['NBYTES_SWEEP'] + 2
            sweeps_scanned = (next_header[1] - current_frame_number) & 0xff
            assert sweeps_scanned == 1  # Debug

        # Second case: the next_header does not match expectations
        else:  # Drop this sweep as next_header does not match expectations
            if verbose:
                print('[WARNING] Lost track of sweep starting at {} '.format(index))
                print('[WARNING] Next header at {} read: {} | Expected: ({}, {})'
                      .format(index+s['NBYTES_SWEEP'], next_header, s['start'], (current_frame_number+1)&0xff))
            flag_success = False

            # Option 1: Look for the start of the correct next_header in the dropped data
            for jj in range(s['NBYTES_SWEEP'])[::-1]:  # Go in reverse
                if jj == s['NBYTES_SWEEP'] - 1:
                    # if sweep_data[jj] == s['start'] and next_frame_number == (current_frame_number+1)&0xff:  # I think this is wrong
                    if batch[jj] == s['start'] and next_header[0] == (current_frame_number + 1) & 0xff:  # Check this
                        next_header = [batch[jj], next_header[0]]
                        # raise
                        break
                    else:
                        next_header = []
                elif batch[jj] == s['start'] and batch[jj + 1] == (current_frame_number + 1) & 0xff:
                    next_header = [batch[jj], batch[jj+1]]
                    break
            # Restart from that location
            if len(next_header):
                if verbose:
                    print("[WARNING] Found header {} at {}".format(next_header, index+jj))
                    print("[WARNING] Skipping sweep {} from {} to {}.".format(current_frame_number, index, index+s['NBYTES_SWEEP']))
                    print("[WARNING] Restarting with sweep {} from position {}".format(next_header[1], index+jj+2))
                index += jj + 2 # Skip the next_header and get ready to scoop s['NBYTES_SWEEP'] of data
                sweeps_scanned = (next_header[1] - current_frame_number) & 0xff
                assert sweeps_scanned == 1  # Debug
            else:
                # Option 2: Search for the header forward
                print("[WARNING] Dropping sweep as the next header was not found in previous data.")
                index, next_header = find_start_batch(data, s, initial_index=index)
                print("[WARNING] Current sweep number: {} | Next header count: {}"
                      .format(current_frame_number, next_header[1]))
                if index == -1:
                    # Failed to find a valid header
                    index = len(data) - 1  # No rest will be generated
                    next_header = [0, current_frame_number]  # Communicate
                    sweeps_scanned = 0  # Skip channel assignement
                    print('[ERROR] Next header not found in previous incorrect frame nor rest of frame.')
                else:
                    sweeps_scanned = (next_header[1] - current_frame_number) & 0xff  # Could be > 1

         # 1 for the backward pass, >= 1 for fwd
        current_frame_number = next_header[1]  # Ready to read the next sweep

        # 3.3 Append data if we are not decimating
        #if sweep_count == s['sweeps_to_drop']:
        for _ in range(sweeps_scanned):
            if sweep_count == s['sweeps_to_drop']:
                if flag_success:
                    signed_data = np.frombuffer(batch, dtype=np.int16)  # Read as int16
                else:
                    signed_data = skipped_data # Will append just 0s. Frame is skipped
                    batch_ch['skipped_sweeps'].append(global_sweep_counter)
    
                for channel in range(s['channel_count']):  # Channels number are 1 based for now
                    batch_ch[channel + 1].append(signed_data[channel::s['channel_count']])  # Data is entangled
    
                # Increment counters
                sweep_count = 0
                global_sweep_counter += 1  # Only count sweeps after decimation
            else: # Decimate sweep: should we interpolate the data or drop zeros?
                if verbose:
                    print("[INFO] Decimating sweep : {}/{}".format(sweep_count, s['sweeps_to_drop']))
                sweep_count += 1
        sweeps_scanned = 0


    # 4. Finalization
    # Return numpy arrays rather than lists
    for k in range(s['channel_count']):
        batch_ch[k + 1] = np.array(batch_ch[k + 1], dtype=np.int16)
    rest = data[index:]  # Get the rest
    if verbose:
        print("\n[INFO] There is a rest of length", len(rest))
    
    return batch_ch, next_header, rest, global_sweep_counter, sweep_count


def calculate_if_data(channel_data, s):
    assert type(channel_data) == dict
    if_data = {}
    clim = s['MAX_DIFFERENTIAL_VOLTAGE']
    for channel in channel_data: # Go through all available channels
        data = np.array(channel_data[channel], dtype=np.float)
        data *= 1 / (s['fir_gain'] * 2 ** (s['adc_bits'] - 1))  # No w
        if_data[channel] = data
    return if_data, clim  # dict


def calculate_angle_plot(sweeps, s, clim, tfd_angles):
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
    if clim is None:
        max_range_i = np.searchsorted(d, s['max_range'])
        clim = np.max(20 * np.log10(np.abs(fx[:max_range_i, :]))) + 10
    if fxm is None:
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


    return fxdb, clim


def calculate_range_time(ch, s, single_sweep=-1):
    # WARNING: ONLY USING CHANNEL 2 FOR THAT
    # TO DO: TAKE THE AVERAGE OF ALL CHANNELS
    # Take the average of all channels
    sweeps = np.zeros(ch[s['active_channels'][0]].shape)  # There is at least one active channel
    for channel in s['active_channels']:
        sweeps += ch[channel].astype(np.int64)
    sweeps /=  s['channel_count']  # Becomes a single 2D numpy array


    if len(ch[s['active_channels'][0]].shape) == 1:  # Only 1 sweep was given
        #sweeps = ch[2]
        nb_sweeps = 1
        sweep_length = len(sweeps)
        single_sweep = True
    else:  # Extract a sweep from an ndarray of them
        if single_sweep != -1:
            #sweeps = sweeps[2][single_sweep]
            sweeps = sweeps[single_sweep]  # Using last sweep only
            nb_sweeps = 1
            sweep_length = len(sweeps)
            single_sweep = True
        else:
            #sweeps = ch[2]
            nb_sweeps = sweeps.shape[0]  # Number of sweeps
            sweep_length = sweeps.shape[1]  # Length of the sweeps
            single_sweep = False
    fourier_len = sweep_length / 2

    """[TBR] Potentially subtract the background & all"""
    subtract_background = False
    subtract_clutter = False

    if subtract_background:
        background = []
        for i in range(sweep_length):
            x = 0
            for j in range(len(sweeps)):
                x += sweeps[j][i]
            background.append(x / len(sweeps))

    max_range_index = int(
        (4 * s['bw'] * fourier_len * s['max_range']) / (s['c'] * s['if_amplifier_bandwidth'] * s['t_sweep']))
    max_range_index = min(max_range_index, sweep_length // 2)
    #print("Max range index:", max_range_index)
    im = np.zeros((max_range_index - 2, nb_sweeps))
    w = [1]*sweep_length if single_sweep else np.kaiser(sweep_length, s['kaiser_beta'])
    m = 0

    for e in range(nb_sweeps):
        sw = sweeps if single_sweep else sweeps[e]
        if subtract_clutter and e > 0:
            sw = [sw[i] - sweeps[e - 1][i] for i in range(sweep_length)]
        if subtract_background:
            sw = [sw[i] - background[i] for i in range(sweep_length)]

        sw = [sw[i] * w[i] for i in range(len(w))]  # Take a Kaiser window of the sweep
        fy = np.fft.rfft(sw)[3:max_range_index + 1]  # FFT of the sweep
        fy = 20 * np.log10((s['adc_ref'] / (2 ** (s['adc_bits'] - 1) * s['fir_gain'] * max_range_index)) * np.abs(fy))
        fy = np.clip(fy, -100, float('inf'))
        m = max(m, max(fy))  # Track max value for m
        im[:, e] = np.array(fy)
        #im[:, e] = fy
        im = np.array(fy) if single_sweep else im

    return im, nb_sweeps, max_range_index, m


def find_start(f, start, s):
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
                f.read(s['NBYTES_SWEEP'])  # Read a whole sweep
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
    #print("START", start)
    sweep_count = 0
    #sweep_count_2 = 0
    signal = s['start']
    global_sweep_counter = 0
    #counter_sweeps_2 = 0
    #skipped_frame_data = np.zeros((s['NBYTES_SWEEP']//4,), dtype=np.int16)

    current_frame_number = first_frame
    # The data will be stored in a dict of lists, the keys being the channel number
    data = dict()
    #data_2 = dict()
    for k in range(s['channel_count']):
        data[k+1] = []
        #data_2[k + 1] = []
    data['skipped_sweeps'] = []  # Stores the skipped frames for both channels
    #data_2['skipped_sweeps'] = []  # Stores the skipped frames for both channels

    while samples == None or i < samples:
        # Read the start signal and the the frame_number
        if verbose:
            print("\n[INFO] Current header [{}, {}] | sweep_data starting at: {}".format(signal, current_frame_number, f.tell()))
        t0 = time.perf_counter()
        sweep_data = f.read(s['NBYTES_SWEEP'])  # Block read
        t1 = time.perf_counter()
        if len(sweep_data) != s['NBYTES_SWEEP']:
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
            for jj in range(s['NBYTES_SWEEP'])[::-1]:  # Go in reverse
                if jj == s['NBYTES_SWEEP']-1:
                    if sweep_data[jj] == s['start'] and signal == (current_frame_number + 1) & 0xff:  # Check this if getting the chance
                        flag_success = True
                        break
                elif sweep_data[jj] == s['start'] and sweep_data[jj+1] == (current_frame_number+1)&0xff:
                    flag_success = True
                    break

            if flag_success:  # The next header was found in the previous sweep data: some data was dropped!
                if verbose:
                    print("[WARNING] Found next header [{}, {}] at position {} in the sweep_data of length {}".format(sweep_data[jj], sweep_data[jj+1], jj, s['NBYTES_SWEEP']))
                # Sanity check:
                f.seek(pos - 2 - s['NBYTES_SWEEP'] + jj)
                signal, next_frame_number = f.read(2)  # Drop previous sweep
                if verbose:
                    print('[WARNING] Jumped to {}, moved by {} byte'.format(f.tell(), f.tell()-pos))
                    print("[WARNING] Skipping sweep {}. New header: [{}, {}] (overall sweep count: {})".format(current_frame_number, signal, next_frame_number, global_sweep_counter))
                # Process the new location
                current_frame_number = next_frame_number

                if f.tell()-pos > 0:
                    # Somehow the previous frame was s['NBYTES_SWEEP'] and did not contain an issue
                    raise ValueError("[ERROR] Why was a correct header not found in the previous data?")

            else:
                raise ValueError('[ERROR] Next header not found in previous incorrect frame. Where is it?')
        else:
            current_frame_number = next_frame_number


        if sweep_count == s['sweeps_to_drop']:
            if verbose:
                print("[INFO] Using this sweep : {}/{}".format(sweep_count, s['sweeps_to_drop']))
            t0 = time.perf_counter()
            signed_data = np.frombuffer(sweep_data, dtype=np.int16)  # Does everything at once
            t1 = time.perf_counter()

            if restart:
                if verbose:
                    print("[WARNING] Due to restart, appending zeros for sweep {} (overall sweep counter: {})".format(current_frame_number-1, global_sweep_counter))
                signed_data = np.zeros((s['NBYTES_SWEEP']//2,), dtype=np.int16)
                data['skipped_sweeps'].append(global_sweep_counter)
            # Append channel data to respective list
            for channel in range(s['channel_count']): # Channels number are 1 based for now
                data[channel+1].append(signed_data[channel::s['channel_count']])
            #print(data[1][0][:10])
            #print(data[2][0][:10])
            #assert not np.array_equal(data[1], data[2])
            global_sweep_counter += 1
            sweep_count = 0
        else: # Decimate sweep: should we interpolate the data or drop zeros?
            if verbose:
                print("[INFO] Decimating sweep : {}/{}".format(sweep_count, s['sweeps_to_drop']))
            sweep_count += 1

    # Return numpy arrays rather than lists
    for k in range(s['channel_count']):
        data[k+1] = np.array(data[k+1], dtype=np.int16)
    return data


def compare_ndarrays(a, b):
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
    zero_sweep = np.zeros((channel_data.shape[1],), dtype=np.int16)
    # Subtract background from the channels data, which are numpy arrays
    # Input is the data dictionnary, output is the list of processed data
    background = np.sum(channel_data, axis=0)/(channel_data.shape[0]-len(data['skipped_sweeps']))
    channel_data = w*(channel_data - background)
    for skipped_frame in data['skipped_sweeps']:
        channel_data[skipped_frame] = zero_sweep
    return channel_data


def subtract_clutter(channel_data, w, data, clutter_averaging=1):
    zero_sweep = np.zeros((channel_data.shape[1],), dtype=np.int16)
    #clutter_averaging = 10  # Number of previous sweeps to use for subtraction
    a = np.zeros((clutter_averaging, channel_data.shape[1]))  # Padding for the first clutter_averaging sweeps
    print(a.shape, channel_data.shape)

    print(sorted(data['skipped_sweeps']))

    # Build the indexes to use for the clutter subtraction
    last_good_indexes = collections.deque(maxlen=clutter_averaging)  # Deque stored the last meaningful sweeps
    last_good_indexes.append(0)  # Assume the first sweep is okay
    subtract_channel_data = []
    #subtract_ch2 = []
    for sweep_number in range(len(channel_data)):
        if sweep_number in sorted(data['skipped_sweeps']):
            subtract_channel_data.append(zero_sweep)
            #subtract_ch2.append(zero_sweep)
        else:
            accumulator_channel_data = zero_sweep.astype(np.float64)  # Temporarily switching to the float space
            #accumulator_ch2 = zero_sweep.astype(np.float64)

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
                #accumulator_ch2 += weights[index]*ch2[ii]
            # if len(last_good_indexes):
                # print("[WARNING] Divide by zero at ", ii)
            accumulator_channel_data /= np.sum(weights)  # Divide by the sum of the weights used
            #accumulator_ch2 /= np.sum(weights)
            # Sanity check, verify that casting back to np.int16 is seamless
            assert np.max(accumulator_channel_data) < 2 ** 16
            #assert np.max(accumulator_ch2) < 2 ** 16
            assert np.min(accumulator_channel_data) > -2 ** 16
            #assert np.min(accumulator_ch2) > -2 ** 16
            subtract_channel_data.append(accumulator_channel_data)
            #subtract_ch2.append(accumulator_ch2)
            last_good_indexes.append(sweep_number)
    # Cast back to np.int16
    subtract_channel_data = np.array(subtract_channel_data, dtype=np.int16)
    print(subtract_channel_data.shape)
    #subtract_ch2 = np.array(subtract_ch2, dtype=np.int16)
    #print(subtract_ch2.shape)
    channel_data = w*(channel_data - subtract_channel_data)
    #ch2 = w*(ch2 - subtract_ch2)
    #assert np.array_equal(channel_data_2[0], w*channel_data[0])  # Verify that the first item did not get subtracted anything
    return channel_data


class Writer(Thread):
    def __init__(self, queue, s, encoding='latin1'):
        Thread.__init__(self)

        self.queue = queue
        self.encoding = encoding
        self.wrote = 0

        if self.encoding == 'latin1':
            with open(s['path_raw_log'], 'w') as f:  # Write the settings to file
                writer = csv.DictWriter(f, fieldnames=s.keys())
                writer.writeheader()
                writer.writerow(s)
            print("[INFO] Opened {} for writing as {} file".format(s['path_raw_log'], self.encoding))
            self.f = open(s['path_raw_log'], 'a', encoding=self.encoding)
        elif self.encoding == 'csv':
            with open(s['path_csv_log'], 'w') as f:
                writer = csv.DictWriter(f, fieldnames=s.keys())
                writer.writeheader()
                writer.writerow(s)
            print("[INFO] Opened {} for writing as {} file".format(s['path_csv_log'], self.encoding))
            self.f = open(s['path_csv_log'], 'a')
            self.writer = csv.writer(self.f)
            self.writer.writerow('')
            self.writer.writerow(['Timestamp', 'Sweep number', 'Channel'])
        else:
            raise ValueError('[ERROR] File encoding method {} unknown'.format(self.encoding))
        self.timeout = s['timeout']

        # Write the settings to the file that was just open


    def run(self):
        while True:
            try:
                d = self.queue.get(True, self.timeout)
            except Empty:
                if self.encoding == 'csv':
                    print('[WARNING] {} Writer timedout after {} s without data | Wrote {} numbers to file'
                          .format(self.encoding, self.timeout, self.wrote))
                elif self.encoding == 'latin1':
                    print('[WARNING] {} Writer timedout after {} s without data | Wrote {} byte'
                          .format(self.encoding, self.timeout, self.wrote))
                self.f.close()
                return
            if len(d) == 0:
                print('\n[INFO] Done after writing {:,} byte'.format(self.wrote))
                self.f.close()
                return
            else:
                if self.encoding == 'latin1':
                    self.f.write(d)
                elif self.encoding == 'csv':
                    self.writer.writerow(d)
                self.wrote += len(d)


class Decode(Thread):
    def __init__(self, queue, s, tfd_angles):
        Thread.__init__(self)

        #t = tfd_angles[0]
        # Initialize the channel data
        self.rest = bytes("", encoding=s['ENCODING'])  # Start without a rest
        self.global_sweep_counter = 0
        self.next_header = [0, 0]
        self.sweep_count = 0

        # IO metrics
        self.received = 0
        self.sent = 0

        # Saving stuff for the run method
        self.s = s  # Need to keep a copy of the settings for the run loop
        self.tfd_angles = tfd_angles

        # Link the input queue
        self.raw_usb_to_decode = queue  # input queue
        self.timeout = s['timeout']

        # Write decoded batches to file
        self.decoded_data_to_file = Queue()  # Output queue
        self.write_decoded_to_file = Writer(self.decoded_data_to_file, s, encoding='csv')
        self.write_decoded_to_file.start()

        # Spawn up to three new processes for the real time display
        # These processes take a single sweep as input and add it to current displays
        # Create the display objects


        # Spawn the process: the update_plot method will be called


    def run(self):
        if_window = display.if_time_domain_animation(self.tfd_angles, self.s, grid=True)
        angle_window = display.angle_animation(self.tfd_angles, self.s, method='cross-range')
        max_range_index = int(
            (4 * self.s['bw'] * (self.s['SWEEP_LENGTH'] / 2) * self.s['max_range']) / (
                    self.s['c'] * self.s['if_amplifier_bandwidth'] * self.s['t_sweep']))
        max_range_index = min(max_range_index, self.s['SWEEP_LENGTH'] // 2)
        range_time_window = display.range_time_animation(self.s, max_range_index)

        t0 = time.perf_counter()

        while True:
            # Get from queue
            try:
                new_batch = self.raw_usb_to_decode.get(True, self.timeout)
                self.received += len(new_batch)
            except Empty:
                print('[WARNING] Decode timedout after {} s without data'.format(self.timeout))
                self.write_decoded_to_file.join()
                return

            if len(new_batch) == 0:
                print('\n[INFO] Decode is done after sending {:,} sweeps to the csv writer'.format(self.sent))
                self.write_decoded_to_file.join()
                return
            else:
                # For now, the processes are started and stopped in here
                # I. Decode that new batch
                if type(new_batch) == str:
                    new_batch = bytes(new_batch, encoding=self.s['ENCODING'])
                batch_ch, next_header, rest, new_counter_sweeps, sweep_count = process_batch(self.rest, new_batch, self.s, self.next_header, self.sweep_count, self.global_sweep_counter, verbose=False)
                self.next_header = next_header
                self.rest = rest
                self.sweep_count = sweep_count


                # II. Send that new batch to be written on the csv file
                for index in range(len(batch_ch[self.s['active_channels'][0]])):  # There is at least one channel
                    for channel in self.s['active_channels']:  # All channels
                        row = [self.s['T']*(self.global_sweep_counter + index), self.global_sweep_counter + index, channel] + batch_ch[channel][index].tolist()
                        self.decoded_data_to_file.put(row)
                self.sent += new_counter_sweeps - self.global_sweep_counter

                print("Sweep: {} | Issued after: {:.3f} s | Decode timestamp: {:.3f} s"
                      .format(self.global_sweep_counter, self.global_sweep_counter*self.s['T'], time.perf_counter()-t0))


                # III. Start all processes concurrently - hopefully they run on different CPUs
                for sweep_counter in range(self.global_sweep_counter, new_counter_sweeps):
                    if sweep_counter%self.s['refresh_stride'] == 0:
                        time_stamp = (None,
                                      self.s['T'] * sweep_counter,
                                      int(self.s['T'] * sweep_counter),
                                      int(1000 * (self.s['T'] * sweep_counter - int(self.s['T'] * sweep_counter))))

                        sweep_to_display = {channel: batch_ch[channel][sweep_counter-self.global_sweep_counter]
                                       for channel in self.s['active_channels']}  # Isolate the sweep

                        # IF TIME DOMAIN
                        #process_if_display = mp.Process(target=if_window.refresh_data, args=(sweep_to_display, self.s, time_stamp[1]))
                        # process_angle_display = mp.Process(target=angle_window.update_plot, args=(last_sweep, ts, clim))
                        # process_range_time_display = mp.Process(target=range_time_window.update_plot, args=(last_sweep, ts, clim))

                        if_data, clim = calculate_if_data(sweep_to_display, self.s)
                        if_window.update_plot(if_data, time_stamp[1], 0)
                        #process_if_display.start()

                        # ANGLE PLOT
                        clim = None
                        fxdb, clim = calculate_angle_plot(sweep_to_display, self.s, clim, self.tfd_angles)
                        #process_angle_display.start()
                        #angle_window.update_plot(fxdb, time_stamp[1], clim)

                        # RANGE TIME
                        im, nb_sweeps, max_range_index, clim = calculate_range_time(sweep_to_display, self.s,
                                                                                                single_sweep=0)
                        range_time_window.update_plot(im, time_stamp[1], clim)
                        #process_range_time_display.start()

                        # JOIN ALL PROCESSES
                        #process_if_display.join()
                        #process_angle_display.join()
                        #process_range_time_display.join()
                self.global_sweep_counter = new_counter_sweeps