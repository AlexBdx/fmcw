from queue import Queue
import datetime
import time
import argparse
import os
import numpy as np
import multiprocessing as mp

from fmcw import *


def main():
    """The main radar function

    This function performs two main operations:
    - I. First, it sets up:
        1. the ADC and the FPGA
        2. Create the time, frequency, distance, angle bases for the plots
        3. Set up the multiprocessing objects
    - II. Second, for a set duration it runs the main loop which consists in:
        1. Read a batch from the FPGA
        2. Push it to a thread that writes it "as is" to file
        3. Decode the batch and:
            1. Push it to a thread that writes it to a csv file. This is human readable
            2. Send that to sub-processes that display:
                1. The IF (Intermediate Frequency) signal of each channel
                2. The angular data
                3. The range time plot

    If you counted well, there is the main process, 2 threads and 3 sub processes total.
    On my to do list: once the loop exits, plot the data from the csv file for a general overview of the run.

    :return: Void
    """
    """I. Set up the subprocesses"""
    launched_processes = []
    flag_reading_data = mp.Event()  # Signals when data is being read. Used to sync camera start if present

    # I.0. [OPTION] Start a camera process that will record in parallel
    if s['record_video']:
        flag_camera_ready = mp.Event()
        camera_process = camera.camera(flag_camera_ready, flag_reading_data, s)
        camera_process.start()
        launched_processes.append(camera_process)
        flag_camera_ready.wait()  # Wait until camera is ready - can take some time to load opencv

    # I.1. Create a subprocess that will read the FPGA
    parent_end, child_end = mp.Pipe()  # Duplex pipe
    fpga = preprocessing.fpga_reader(flag_reading_data, child_end, s)
    process_info = parent_end.recv()
    print("[INFO] FPGA reader's PID is", process_info['pid'])
    fpga.start()  # Start the reader right away
    launched_processes.append(fpga)

    # I.2. Create all the bases needed for the plots
    t, f, d, angles = postprocessing.create_bases(s)
    angle_mask = ~(np.isnan(angles) + (np.abs(angles) > s['angle_limit']))
    print("Max ADC range: {:.1f} m".format(d[-1]))
    tfd_angles = (t, f, d, angles, angle_mask)

    # I.3. Multiprocessing related objects
    # Write settings to file
    settings_to_file = Queue()
    write_settings_to_file = postprocessing.Writer(settings_to_file, s, encoding='settings')  # Spawn a new thread
    del write_settings_to_file  # The settings are written during __init__ so no need to run

    # Write raw data directly to file
    raw_usb_data_to_file = Queue()
    write_raw_to_file = postprocessing.Writer(raw_usb_data_to_file, s, encoding='latin1')  # Spawn a new thread
    write_raw_to_file.start()  # Starts it on a separate thread
    launched_processes.append(write_raw_to_file)

    # Write decoded data to csv
    decoded_data_to_file = Queue()  # Output queue
    write_decoded_to_file = postprocessing.Writer(decoded_data_to_file, s, encoding='csv')  # Spawn a new thread
    write_decoded_to_file.start()
    launched_processes.append(write_decoded_to_file)

    # Initialize the variables that we carry over batch after batch
    rest = bytes("", encoding=s['encoding'])  # Start without a rest
    sweep_count = 0  # Overall number of sweeps processed
    next_header = [0, 0]  # Signals an invalid header to trigger a find_start
    counter_decimation = 0  # Rolling counter, keeps track of software decimation across batches

    # Create shared variable to communicate with the sub-processes
    sweep_displayed = mp.Array('i', s['nb_values_sweep'], lock=False)  # 1D, shared, static array which is zeroed
    time_stamp = mp.Array('d', 4, lock=False)  # TO DO: some entries might be unused
    data_accessible = mp.Event()  # clear when updating, set when data is accessible. Acts more or less as a Lock
    data_accessible.set()
    new_sweep_if = mp.Event()  # set when there is new data available, clear when still the same sweep
    new_sweep_angle = mp.Event()
    new_sweep_range = mp.Event()

    # Spawn 3 sub-processes for figure display
    process_if = postprocessing.if_display(tfd_angles, s,
                                           data_accessible, new_sweep_if, sweep_displayed, time_stamp)
    process_angle = postprocessing.angle_display(tfd_angles, s,
                                                 data_accessible, new_sweep_angle, sweep_displayed, time_stamp)
    process_range_time = postprocessing.range_time_display(tfd_angles, s,
                                                           data_accessible, new_sweep_range, sweep_displayed, time_stamp)
    # Start all 3 sub-processes
    process_angle.start()
    process_range_time.start()
    process_if.start()
    launched_processes.append(process_angle)
    launched_processes.append(process_range_time)
    launched_processes.append(process_if)


    """II. Data Acquisition"""
    # This bloc makes sure that all the processes initialized correctly
    def check_processes(process_list, timeout=s['timeout']):
        """
        Checks if all processes in a list are alive.
        :param process_list: List of processes
        :param timeout: Maximum duration of the test loop
        :return: True is all of them are alive, False if at least one is dead
        """
        t_initial = time.perf_counter()
        flag_success = False
        while time.perf_counter()-t_initial < timeout and not flag_success:
            flag_success = True
            for process in process_list:
                if not process.is_alive():
                    flag_success = False
                    break
        return flag_success
    if not check_processes(launched_processes):
        for process in launched_processes:
            if not process.is_alive():  # There is a dead process -> terminate everything that is still alive
                print("Process", process, "is dead")
                for p in launched_processes:
                    try:
                        p.terminate()
                    except:
                        pass
                raise OSError('[ERROR] Process {} is dead. Aborting.'.format(process))

    timing = []
    counter_zero = 0
    try:
        t0 = time.perf_counter()  # Keep track of starting time
        while time.perf_counter() - t0 < s['duration']:  # Endless if np.inf
            # II.1. Read a batch
            t1 = time.perf_counter()
            # raw_usb_data = fpga.device.read(s['byte_usb_read'])
            # raw_usb_data = raw_usb.get(True, s['timeout'])  # A subprocess reads the data from the FPGA
            raw_usb_data = b""
            for _ in range(s['sub_call']):
                raw_usb_data += parent_end.recv_bytes()
            timing.append(time.perf_counter()-t1)

            if len(raw_usb_data) != 0:
                # II.2. Write the binary data to file
                raw_usb_data_to_file.put(str(raw_usb_data))
                if type(raw_usb_data) == str:  # If device is opened in text mode
                    raw_usb_data = bytes(raw_usb_data, encoding=s['encoding'])

                # II.3. Process the batch
                batch_ch, next_header, rest, new_sweep_count, counter_decimation = postprocessing.process_batch(rest, 
                                                                                             raw_usb_data,
                                                                                             s,
                                                                                             next_header,
                                                                                             counter_decimation,
                                                                                             sweep_count,
                                                                                             verbose=False)

                counter_zero += len(batch_ch['skipped_sweeps'])  # Count how many zeros were added
                # II.3.1. Send that new batch to be written to the csv file
                for index in range(len(batch_ch[s['active_channels'][0]])):  # There is at least one channel
                    for channel in s['active_channels']:  # All channels
                        row = {
                            'ts': s['T'] * (sweep_count + index),
                            'sweep_count': sweep_count + index,
                            'channel': channel,
                            'data': batch_ch[channel][index]
                        }  # Better for readability
                        decoded_data_to_file.put([row['ts'], row['sweep_count'], row['channel'], *row['data'].tolist()])

                # II.3.2. Refresh the plots via shared variables
                for index_sweep in range(sweep_count, new_sweep_count):
                    if index_sweep % s['refresh_stride'] == 0:
                        # /!\ SHARED MEMORY UPDATE /!\
                        data_accessible.clear()  # Lock the data during update

                        # 1. Update the time_stamp entries
                        time_stamp[0] = index_sweep
                        time_stamp[1] = s['T'] * index_sweep
                        time_stamp[2] = time.perf_counter() - t0  # Time since data was sent by FPGA
                        time_stamp[3] = int(1000 * (s['T'] * index_sweep - int(s['T'] * index_sweep)))

                        # 2. Copy the relevant sweep data to the shared array
                        sweep_data = [batch_ch[channel][index_sweep - sweep_count]
                                      for channel in s['active_channels']]  # Aggregate all channels
                        sweep_data = np.array(sweep_data, dtype=np.int16)  # Make the list of list 2D numpy
                        sweep_data = sweep_data.reshape((-1,))  # Make it 1D for the static array
                        assert len(sweep_data) == len(sweep_displayed)
                        for entry in range(len(sweep_data)):
                            sweep_displayed[entry] = sweep_data[entry]  # Brute force copy to shared array

                        # 3. Indicate that new data is available too each sub-process
                        new_sweep_if.set()
                        new_sweep_angle.set()
                        new_sweep_range.set()
                        data_accessible.set()

                sweep_count = new_sweep_count

    # 4. Close properly all objects when done
    finally:
        # 1. Close the FMCW device
        fpga.terminate()
        flag_reading_data.clear()
        if s['record_video']:
            flag_camera_ready.clear()  # This will shut make the process return hence terminate

        # 2. Close the queues
        raw_usb_data_to_file.put('')
        decoded_data_to_file.put('')
        # 3. Join with the writer once they timeout
        write_raw_to_file.join()
        write_decoded_to_file.join()
        # 4. Close the plotting processes
        process_if.terminate()
        process_angle.terminate()
        process_range_time.terminate()

    print("[INFO] Total sweeps: {} | Zeros: {} | Ratio: {:.1f} %"
          .format(sweep_count, counter_zero, 100*(1-counter_zero/sweep_count)))
    print("[INFO] Read timing results: mean = {:.6f} s, std = {:.6f} s".format(np.mean(timing), np.std(timing)))


"""------------------------------------------------------------------------------------------------------------------"""
"""I. Parameters setup"""npt
# I.1. Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--duration", type=int, default=20, help="duration of the recording [s]")
ap.add_argument("-f", "--log_folder", type=str, default=os.path.join(os.getcwd(), 'Recordings'), help="Path to folder containing the output logs")
args = vars(ap.parse_args())
duration = args["duration"]
if duration == 0:
    duration = np.inf  # Will run endlessly until keyboard interrupt
path_log_folder = args["log_folder"]
encoding = 'latin1'
binary = True

# I.2. [USER] USER PARAMETERS: FEEL FREE TO EDIT
s_gen = {
    'duration': duration,
    'byte_usb_read': 0x1000,
    'sub_call': 1,
    'bw': 600e6,
    't_sweep': 2e-3,
    't_delay': 1e-3,
    'down_sampler': True,
    'quarter': False,
    'acquisition_decimate': 2,
    'soft_decimate': 0,
    'max_range': 5000,
    'refresh_period': 0.125,
    'range_time_to_display': 10,
    'real_time_recall': max(duration//5, 1),
    'subtract_background': False,
    'subtract_clutter': 0,
    'flag_Hanning': False,
    'record_video': False
}

active_channels = {
    1: True,
    2: True,
}  # Currently there are only 2 physical channels. Could be increased with the multiplexer

# I.3. [USER] MODIFY ONLY IF YOU KNOW WHAT YOU ARE DOING
ts = datetime.datetime.now().strftime('%y%m%d_%H%M%S_')
s_tech = {
    'kaiser_beta': 6,
    'channel_offset': 21,
    'swap_chs': True,
    'pa_off_advance': 0.2e-3,
    'encoding': encoding,
    'timeout': 1,
    'patience_valid_header': 10,
    'path_settings': os.path.join(path_log_folder, ts+'settings.csv'),
    'path_raw_log': os.path.join(path_log_folder, ts+'fmcw3.log'),
    'path_csv_log': os.path.join(path_log_folder, ts+'fmcw3.csv')
}

s_camera = {
    'camera_address': 0,
    'fps': 10,
    'camera_resolution': (640, 480),
    'path_video': os.path.join(path_log_folder, ts+'output.avi')
}

s_display = {
    'cblim_if': [-1, 1],
    'cblim_angle': [40, 80],
    'cblim_range_time': [-80, 0],
}

# I.4. [FROZEN] DESIGN & CALCULATED PARAMETERS. THEY SHOULD NOT BE MODIFIED
s_hw = {
    'c': 299792458.0,
    'f0': 5.3e9,
    'if_amplifier_bandwidth': 2e6,
    'fir_gain': 9.0,
    'adc_ref': 1,
    'adc_bits': 12,
    'd_antenna': 28e-3,
    'angle_limit': 55,
    'angle_pad': 100,
    'max_differential_voltage': 1,
    'start': b'\x7f'[0]
}
if s_gen['down_sampler']:
    s_hw['if_amplifier_bandwidth'] /= 2
    if s_gen['quarter']:
        s_hw['if_amplifier_bandwidth'] /= 2

s_calc = {
    'channel_count': sum(active_channels.values()),
    'active_channels': [key for key in active_channels if active_channels[key]],
    'adc_bytes': s_hw['adc_bits']//8 + 1,
    'range_adc': s_hw['c']*s_hw['if_amplifier_bandwidth']/(2*s_gen['bw']/s_gen['t_sweep']),
    'sweep_length': int(s_gen['t_sweep'] * s_hw['if_amplifier_bandwidth']),
    'overall_decimate': (s_gen['soft_decimate'] + 1) * (s_gen['acquisition_decimate'] + 1),
}
s_calc['T'] = (s_gen['t_sweep'] + s_gen['t_delay']) * s_calc['overall_decimate']
s_calc['refresh_stride'] = round(s_gen['refresh_period']/s_calc['T']) if s_gen['refresh_period'] >= s_calc['T'] else 1
s_calc['nbytes_sweep'] = s_calc['channel_count']*s_calc['sweep_length']*s_calc['adc_bytes']  # in byte
s_calc['nb_values_sweep'] = s_calc['channel_count']*s_calc['sweep_length']  # length of a 1D array containing a sweep
s_calc['patience_data_length'] = s_tech['patience_valid_header']*s_calc['nbytes_sweep']

# Merge all dictionaries together
s_temporary = {**s_gen, **s_camera, **active_channels, **s_tech, **s_display, **s_hw, **s_calc}


# I.5. Make the settings dictionary read only
class ReadOnlyDict(dict):  # Settings should be read only, so the final dict will be read only
    __readonly = False  # Start with a read/write dict

    def set_read_state(self, read_only=True):
        """Allow or deny modifying dictionary"""
        self.__readonly = bool(read_only)

    def __setitem__(self, key, value):
        if self.__readonly:
            raise RuntimeError('This dictionary is currently read only!')
        return dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        if self.__readonly:
            raise RuntimeError('This dictionary is currently read only!')
        return dict.__delitem__(self, key)


# Transfer s to a read only dict
read_only_dict = ReadOnlyDict()
for key in s_temporary:  # Brute force copy
    read_only_dict[key] = s_temporary[key]
s = read_only_dict  # Copy back
s.set_read_state(read_only=True)  # Set as read only


"""II. Run the main loop"""
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # starts a fresh python interpreter process
    # Try to max out the priority of the process
    try:
        print("[INFO] Process set to niceness", os.nice(-20))  # Only root processes can be below 0
    except PermissionError:
        print("[INFO] Process set to niceness", os.nice(0))  # Highest priority for users



    main()

    """[Display overall range time when done]"""
    # Once the main is over, let's plot the overall range-time shall we?
    path = '/home/alex/Desktop/Helico/5_Radar/src/Recordings'
    #ts = '190820_115214_'
    s = display.import_settings(path, ts)  # Reuse timestamp
    ch = display.import_csv(path, ts, s)
    t = np.linspace(0, s['overall_decimate'] * len(ch[1]) * (s['t_sweep'] + s['t_delay']), len(ch[1]))

    im, nb_sweeps, _ = postprocessing.calculate_range_time(ch, s)

    display.plot_range_time(t, im, s, ts)
    """"""

