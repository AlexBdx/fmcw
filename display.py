import matplotlib
# matplotlib.use('TkAgg')  # Use another backend
import matplotlib.pyplot as plt
import numpy as np
import os
import fmcw.postprocessing as postprocessing # ololololo
import csv


def plot_if_spectrum(d, ch, sweep_number, w, fir_gain, adc_bits, time_stamp, show_plot=False):
    """TO DO: NEEDS AN UPDATE TO REDUCE THE ARGUMENT COUNT VIA THE USE OF THE SETTINGS DICTIONARY
    Plot the IF Fourier spectrum. Useful to see how much noise there is in the data and where
    :param d: [m] distance bins TO DO: why is it here?
    :param ch: Dictionary containing the data
    :param sweep_number: sweep to plot
    :param w: Window to use when processing the data
    :param fir_gain: TO DO: will be superseeded by the settings dictionary
    :param adc_bits: TO DO: will be superseeded by the settings dictionary
    :param time_stamp: [bool] Save the data?
    :param show_plot: [bool] Show the plot?
    :return:
    """
    if time_stamp[0]:
        save_path = os.path.join(time_stamp[0], 'IF_spectrum_{:03d}S{}.png'.format(time_stamp[2], time_stamp[3]))
    ch = {k: v for k, v in ch.items() if type(k) == int}  # Avoid 'skipped_frames' key
    plt.figure()
    for channel in ch:
        fx = ch[channel][sweep_number]
        fx = np.array(fx, dtype=np.float)
        fx *= w/(fir_gain*2**(adc_bits-1))
        fx = 2*np.fft.rfft(fx)/(len(fx))
        fx = 20*np.log10(np.abs(fx))
        plt.plot(d, fx, label='Channel '+str(channel))
    
    plt.legend(loc='upper right')
    plt.title('IF spectrum for sweep {}'.format(sweep_number))
    plt.ylabel("Amplitude [dBFs]")
    plt.xlabel("Distance [m]")
    if time_stamp[0]:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def plot_if_time_domain(fig_if, t, ch, sweep_number, s, ylim, time_stamp, show_plot=False):
    """TO DO: ACTUALIZE THE ARGUMENTS WITH THE NEW METHOD
    Plot the IF data for a bunch of sweeps
    :param fig_if:
    :param t:
    :param ch:
    :param sweep_number:
    :param s:
    :param ylim:
    :param time_stamp:
    :param show_plot:
    :return:
    """
    if time_stamp[0]:
        save_path = os.path.join(time_stamp[0], 'IF_{:03d}S{}.png'.format(time_stamp[2], time_stamp[3]))
    ch = {k: v for k, v in ch.items() if type(k) == int}  # Avoid 'skipped_frames' key
    plt.figure(fig_if)
    #plt.ion()
    for channel in ch:
        if_data = ch[channel][sweep_number]
        if_data = np.array(if_data, dtype=np.float)
        if_data *= 1/(s['fir_gain']*2**(s['adc_bits']-1))  # No w
        plt.plot(t, if_data, label='Channel '+str(channel))

    plt.title('IF time-domain at time T = {:.3f} s'.format(time_stamp[1]))
    plt.ylabel("Voltage [V]")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.ylim(ylim)
    
    plt.legend(loc='upper right')
    if time_stamp[0]:
        plt.savefig(save_path)
    if show_plot:
        plt.show()


def plot_angle(t, d, fxdb, angles_masked, clim, max_range, time_stamp, method='', show_plot=False):
    """TO DO: ACTUALIZE THE ARGUMENTS WITH THE NEW METHOD
    Plot the angular data for a bunch of sweeps
    :param t:
    :param d:
    :param fxdb:
    :param angles_masked:
    :param clim:
    :param max_range:
    :param time_stamp:
    :param method:
    :param show_plot:
    :return:
    """
    plt.ioff()
    if time_stamp[0]:
        save_path = os.path.join(time_stamp[0], 'range_{:03d}S{}.png'.format(time_stamp[2], time_stamp[3]))
    fig = plt.figure()
    if method == 'polar':
        ax = fig.add_subplot(111, polar=True)
        imgplot = ax.pcolormesh(angles_masked*np.pi/180, d, fxdb.transpose())
    elif method == 'cross-range':
        r, t = np.meshgrid(d, angles_masked*np.pi/180)
        x = r*np.cos(t)
        y = -r*np.sin(t)
        imgplot = plt.pcolormesh(x, y, fxdb)
        plt.colorbar()
        ylim = 90*np.sin(angles_masked[0]*np.pi/180)
        #plt.ylim([-ylim, ylim])
        plt.ylim([-30, 30])
        plt.xlim([d[0], max_range])
        plt.ylabel("Cross-range [m]")
    else:
        imgplot = plt.pcolormesh(d, angles_masked, fxdb)
        plt.colorbar()
        plt.ylim([angles_masked[0], angles_masked[-1]])
        plt.xlim([d[0], max_range])
        plt.ylabel("Angle [$^o$]")
    imgplot.set_clim(clim-50,clim)
    plt.title("Angle plot at T = {:.3f} s".format(time_stamp[1]))
    plt.xlabel("Range [m]")
    if time_stamp[0]:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()
    plt.ion()

def plot_range_time(t, im, s, time_stamp=''):
    """TO DO: ACTUALIZE THE ARGUMENTS WITH THE NEW METHOD
    Range time plot of a bunch of sweeps
    :param t:
    :param meshgrid_data:
    :param m:
    :param time_stamp:
    :param show_plot:
    :return:
    """
    plt.ioff()
    max_range_index = int(s['sweep_length'] * s['max_range'] / s['range_adc'])
    max_range_index = min(max_range_index, s['sweep_length'] // 2)
    x, y = np.meshgrid(t, np.linspace(0, s['range_adc']*max_range_index/s['sweep_length'], max_range_index-2))

    if time_stamp:
        save_path = os.path.join(os.path.split(s['path_csv_log'])[0], time_stamp[:-1]+'.png')
    plt.figure()
    plt.ylabel("Range [m]")
    plt.xlabel("Time [s]")
    plt.xlim([t[0], t[-1]])
    plt.title('Overall Range-time plot for '+time_stamp[:-1])
    imgplot = plt.pcolormesh(x, y, im)
    imgplot.set_clim(*s['cblim_range_time'])
    print("[INFO] cblim used", s['cblim_range_time'])
    plt.colorbar()

    if matplotlib.get_backend() == 'Qt5Agg':
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
    elif matplotlib.get_backend() == 'TkAgg':
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
    elif matplotlib.get_backend() == 'WXAgg':
        manager = plt.get_current_fig_manager()
        manager.frame.Maximize(True)
    else:
        raise ValueError('Not sure how to maximize image with this backend')

    if time_stamp:
        plt.savefig(save_path, bbox_inches = 'tight', dpi=300)
    plt.show()



def import_settings(path, timestamp):
    with open(os.path.join(path, timestamp+'settings.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row
        for key in s:
            try:
                s[key] = eval(s[key])
            except NameError:
                print("[WARNING] {} has no interpretation, probably a str setting. No action taken.".format(key))
            except SyntaxError:
                print("[WARNING] {} has no valid interpretation, probably a str setting. No action taken.".format(key))
    return s

def import_csv(path, timestamp, s):
    with open(os.path.join(path, timestamp+'fmcw3.csv')) as f:
        reader = csv.reader(f)
        next(reader)
        data = {channel: [] for channel in s['active_channels']}
        for row in reader:  # Place the rows in the right channel
            data[eval(row[2])].append(row[3:])
        for channel in data:
            data[channel] = np.array(data[channel], dtype=np.int16)
        return data
