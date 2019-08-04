import ast
import numpy as np
import time
import collections

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


def r4_normalize(x, d, e=1.5):
    n = d[-1]**e
    return x*d**e/n


def read_settings(f, encoding=None):
    f.seek(0)
    data = f.readline()
    data = data.decode(encoding) if encoding else data
    settings = ast.literal_eval(data)
    return settings


def find_start(f, start, nbytes_sweep):
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
                f.read(nbytes_sweep)  # Read a whole sweep
                if f.read(1) != start:
                    done = False
                next_frame_number = f.read(1)

                if len(next_frame_number) == 0:
                    return
                if next_frame_number[0] != current_frame_number+1+j:
                    done = False
            f.seek(current_position)  # Go back to just before signal & frame number
    return current_frame_number


def import_data(f, start, first_frame, nbytes_sweep, samples, decimate_sweeps, nb_channels=1):
    #print("START", start)
    sweep_count = 0
    signal = start[0]
    counter_sweeps = 0
    counter_skipped_lines = 0
    skipped_frame_data = np.zeros((nbytes_sweep//4,), dtype=np.int16)

    current_frame_number = first_frame
    # The data will be stored in a dict of lists, the keys being the channel number
    data = dict()
    #data_2 = dict()
    for k in range(nb_channels):
        data[k+1] = []
        #data_2[k + 1] = []
    data['skipped_frames'] = []  # Stores the skipped frames for both channels

    while samples == None or i < samples:
        # Read the start signal and the the frame_number
        print("\n[INFO] Current header [{}, {}] | sweep_data starting at: {}".format(signal, current_frame_number, f.tell()))
        t0 = time.perf_counter()
        sweep_data = f.read(nbytes_sweep)  # Block read
        t1 = time.perf_counter()
        if len(sweep_data) != nbytes_sweep:
            break  # No more data, we have reached the end of the file

        # Read the header
        signal, next_frame_number = f.read(2) # Should get the next signal and frame_number
        restart = False
        if signal != start[0]:
            print('[WARNING] Lost track of start at {} | Next header read: [{}, {}] | Expected header: [{}, {}]'
                  .format(f.tell(), signal, next_frame_number, start[0], (current_frame_number+1)&0xff))
            restart = True
        if restart == False and current_frame_number != None:
            if next_frame_number != (current_frame_number+1)&0xff:
                print('[WARNING] Lost a sweep. Previous {}, now {} at {}'.format(current_frame_number, next_frame_number, f.tell()))
                assert 1==0
                restart = True

        if restart: # Find the nearest start flag, looking at the latest data first
            pos = f.tell()
            # Check in the data if a valid header can be found
            flag_success = False
            for jj in range(nbytes_sweep)[::-1]:  # Go in reverse
                if jj == nbytes_sweep-1:
                    if sweep_data[jj] == start[0] and next_frame_number == (current_frame_number+1)&0xff:
                        flag_success = True
                elif sweep_data[jj] == start[0] and sweep_data[jj+1] == (current_frame_number+1)&0xff:
                    flag_success = True
                
                if flag_success:
                    print("[WARNING] Found header [{}, {}] at position {} in the sweep_data of length {}".format(sweep_data[jj], sweep_data[jj+1], jj, nbytes_sweep))
                    # Sanity check:
                    f.seek(pos - 2 - nbytes_sweep + jj)
                    signal, next_frame_number = f.read(2)  # Drop previous sweep
                    print('[WARNING] Jumped to {}, moved by {} byte'.format(f.tell(), f.tell()-pos))
                    print("[WARNING] Skipping sweep {}. New header: [{}, {}] (overall sweep count: {})".format(current_frame_number, signal, next_frame_number, counter_sweeps))
                    # Process the new location
                    current_frame_number = next_frame_number
                    
                    if f.tell()-pos > 0:
                        # Somehow the previous frame was nbytes_sweep and did not contain an issue
                        raise ValueError("[ERROR] Why was a correct header not found in the previous data?")
                    
                    break
        else:
            current_frame_number = next_frame_number

        if decimate_sweeps <= 1 or sweep_count >= decimate_sweeps:
            # Convert to 2's complement grabbing byte 2 at a time
            t0 = time.perf_counter()
            #signed_data = [twos_comp(sweep_data[2 * ii] + (sweep_data[2 * ii + 1] << 8), 16) for ii in
                           #range(int(nbytes_sweep / 2))]

            signed_data = np.frombuffer(sweep_data, dtype=np.int16)  # Does everything at once
            #signed_data_2 = np.invert(signed_data_2)
            #assert np.array_equal(signed_data, signed_data_2)

            #print(signed_data)
            t1 = time.perf_counter()
            #print(t1-t0)
            if nb_channels == 2:  # Data is entangled: 2 byte for ch1 then 2 byte for ch2
                if restart: # There is no data
                    print("[WARNING] Due to restart, appending zeros for sweep {} (overall sweep counter: {})".format(current_frame_number-1, counter_sweeps))
                    data[1].append(skipped_frame_data)
                    data[2].append(skipped_frame_data)
                    data['skipped_frames'].append(counter_sweeps)
                    #data_2[1].append(skipped_frame_data)
                    #data_2[2].append(skipped_frame_data)
                    counter_skipped_lines += 1
                else:
                    #print("[INFO] max: {}, min: {}".format(np.max(signed_data[::2]), np.min(signed_data[::2])))
                    data[1].append(signed_data[::2])
                    data[2].append(signed_data[1::2])
                    #data_2[1].append(signed_data_2[::2])
                    #data_2[2].append(signed_data_2[1::2])
                    #assert 1==0
                counter_sweeps += 1

            else:
                if restart:
                    data[1].append(skipped_frame_data)
                    counter_skipped_lines += 1
                else:
                    data[1].append(signed_data)
                counter_sweeps += 1
            sweep_count = 0
        else:
            sweep_count += 1
    return data, counter_skipped_lines


def subtract_background(channel_data, w, data):
    zero_sweep = np.zeros((channel_data.shape[1],), dtype=np.int16)
    # Subtract background from the channels data, which are numpy arrays
    # Input is the data dictionnary, output is the list of processed data
    background = np.sum(channel_data, axis=0)/(channel_data.shape[0]-len(data['skipped_frames']))
    channel_data = w*(channel_data - background)
    for skipped_frame in data['skipped_frames']:
        channel_data[skipped_frame] = zero_sweep
    return channel_data


def subtract_clutter(channel_data, w, data, clutter_averaging=1):
    zero_sweep = np.zeros((channel_data.shape[1],), dtype=np.int16)
    #clutter_averaging = 10  # Number of previous sweeps to use for subtraction
    a = np.zeros((clutter_averaging, channel_data.shape[1]))  # Padding for the first clutter_averaging sweeps
    print(a.shape, channel_data.shape)

    print(sorted(data['skipped_frames']))

    # Build the indexes to use for the clutter subtraction
    last_good_indexes = collections.deque(maxlen=clutter_averaging)  # Deque stored the last meaningful sweeps
    last_good_indexes.append(0)  # Assume the first sweep is okay
    subtract_channel_data = []
    #subtract_ch2 = []
    for sweep_number in range(len(channel_data)):
        if sweep_number in sorted(data['skipped_frames']):
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
