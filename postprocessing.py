

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


def f_to_d(f, bw, sweep_length):
    c = 299792458.0
    return c*f/(2*(bw/sweep_length))


def r4_normalize(x, d, e=1.5):
    n = d[-1]**e
    return x*d**e/n


def read_settings(f):
    f.seek(0)
    data = f.readline()
    data = data.decode(ENCODING) if BINARY else data
    settings = ast.literal_eval(data)
    return settings


def find_start(f):
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
                f.read(NBYTES_SWEEP)  # Read a whole sweep
                if f.read(1) != start:
                    done = False
                next_frame_number = f.read(1)

                if len(next_frame_number) == 0:
                    return
                if next_frame_number[0] != current_frame_number+1+j:
                    done = False
            f.seek(current_position)  # Go back to just before signal & frame number
    return current_frame_number


def import_data(f, start, first_frame, nb_channels=1):
    sweep_count = 0
    signal = start[0]
    counter_sweeps = 0
    counter_skipped_lines = 0
    skipped_frame_data = np.zeros((NBYTES_SWEEP//4,), dtype=np.int16)

    current_frame_number = first_frame
    # The data will be stored in a dict of list, the keys being the channel number
    data = dict()
    #data_2 = dict()
    for k in range(nb_channels):
        data[k+1] = []
        #data_2[k + 1] = []
    data['skipped_frames'] = []  # Stores the skipped frames for both channels

    while samples == None or i < samples:
        # Read the start signal and the the frame_number
        print("[INFO] id: {} | frame_number: {} | sweep_data starting at: {}".format(signal, current_frame_number, f.tell()))
        t0 = time.perf_counter()
        #if f.tell() != 202:
            #break
        sweep_data = f.read(NBYTES_SWEEP)  # Block read
        t1 = time.perf_counter()
        #print(t1-t0)
        #j += NBYTES_SWEEP
        #i += NBYTES_SWEEP
        if len(sweep_data) != NBYTES_SWEEP:
            break  # No more data, we have reached the end of the file

    #if j == NBYTES_SWEEP:
        signal, next_frame_number = f.read(2) # Should get the next signal and frame_number
        restart = False
        if signal != start[0]:
            print('[WARNING] Lost track of start at {} | Header read: [{}, {}] | Expected header: [{}, {}]'
                  .format(f.tell(), signal, next_frame_number, start[0], current_frame_number))
            restart = True
        if restart == False and current_frame_number != None:
            if next_frame_number != (current_frame_number+1)&0xff:
                print('[WARNING] Lost a sweep. Previous {}, now {} at {}'.format(current_frame_number, next_frame_number, f.tell()))
                assert 1==0
                restart = True

        if restart: # Find the nearest start flag, looking at the latest data first
            pos = f.tell()

            # First option: Look at the sweep_data array to see if the previous frame was not "too short"
            for jj in range(NBYTES_SWEEP-1, 0, -1):
                if sweep_data[jj] == start[0] and sweep_data[jj+1] == (current_frame_number+1)&0xff:
                    print("[WARNING] Found header [{}, {}] at {} in the sweep_data".format(sweep_data[jj], sweep_data[jj+1], jj))
                    # Sanity check:
                    f.seek(pos - 2 - NBYTES_SWEEP + jj)
                    signal, next_frame_number = f.read(2)  # Start here and take 1 loss
                    print("[WARNING] Skipping sweep {}".format(counter_sweeps))
                    #print("[WARNING] Position {}: {} | Position {}: {}".format(f.tell()-2, signal, f.tell()-1, next_frame_number))
                    #f.seek(pos - 2 - NBYTES_SWEEP + jj)
                    break

            #current_frame_number = find_start(f) # Should be right on a start byte if found in sweep_data
            current_frame_number = next_frame_number
            print('[WARNING] Jumped to {}, moved by {} byte'.format(f.tell(), f.tell()-pos))
            if f.tell()-pos > 0:
                # Somehow the previous frame was NBYTES_SWEEP and did not contain an issue
                raise ValueError("[ERROR] Why was a correct header not found in the previous data?")
            #continue
        else:
            current_frame_number = next_frame_number

        if decimate_sweeps <= 1 or sweep_count >= decimate_sweeps:
            # Convert to 2's complement grabbing byte 2 at a time
            t0 = time.perf_counter()
            #signed_data = [twos_comp(sweep_data[2 * ii] + (sweep_data[2 * ii + 1] << 8), 16) for ii in
                           #range(int(NBYTES_SWEEP / 2))]

            signed_data = np.frombuffer(sweep_data, dtype=np.int16)  # Does everything at once
            #signed_data_2 = np.invert(signed_data_2)
            #assert np.array_equal(signed_data, signed_data_2)

            #print(signed_data)
            t1 = time.perf_counter()
            #print(t1-t0)
            if channels == 2:  # Data is entangled: 2 byte for ch1 then 2 byte for ch2
                if restart: # There is no data
                    data[1].append(skipped_frame_data)
                    data[2].append(skipped_frame_data)
                    data['skipped_frames'].append(counter_sweeps)
                    #data_2[1].append(skipped_frame_data)
                    #data_2[2].append(skipped_frame_data)
                    counter_skipped_lines += 1
                else:
                    print("[INFO] max: {}, min: {}".format(np.max(signed_data[::2]), np.min(signed_data[::2])))
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
