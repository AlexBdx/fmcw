import matplotlib.pyplot as plt
import numpy as np
import os

def plot_range_time(t, meshgrid_data, m, save_path, show_plot=False):
    # Range time plot

    plt.figure()
    plt.ylabel("Range [m]")
    plt.xlabel("Time [s]")
    plt.xlim([t[0], t[-1]])
    plt.title('Range-time plot')
    imgplot = plt.pcolormesh(*meshgrid_data)
    imgplot.set_clim(m-80,m)
    plt.colorbar()
    plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def plot_if_spectrum(d, ch, sweep_number, w, fir_gain, adc_bits, save_path, show_plot=False):
    # IF spectrum plot
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
    plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def plot_if_time_domain(t, ch, sweep_number, fir_gain, adc_bits, ylim, time_stamp, show_plot=False):
    # IF time domain
    save_path = os.path.join(time_stamp[0], 'IF_{:03d}S{}.png'.format(time_stamp[2], time_stamp[3]))
    ch = {k: v for k, v in ch.items() if type(k) == int}  # Avoid 'skipped_frames' key
    plt.figure()
    for channel in ch:
        if_data = ch[channel][sweep_number]
        if_data = np.array(if_data, dtype=np.float)
        if_data *= 1/(fir_gain*2**(adc_bits-1))  # No w
        plt.plot(t, if_data, label='Channel '+str(channel))

    plt.title('IF time-domain at time T = {:.3f} s'.format(time_stamp[1]))
    plt.ylabel("Voltage [V]")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.ylim(ylim)
    
    plt.legend(loc='upper right')
    plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def plot_angle(d, fxdb, angles_masked, clim, max_range, time_stamp, method='', show_plot=False):
    # Angle plots
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
    plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()
    
