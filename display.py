import matplotlib.pyplot as plt
import numpy as np
import os
import fmcw.postprocessing as postprocessing # ololololo
plt.ion()




def plot_if_spectrum(d, ch, sweep_number, w, fir_gain, adc_bits, time_stamp, show_plot=False):
    # IF spectrum plot
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
    # IF time domain
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
    #plt.close()
    



def plot_angle(t, d, fxdb, angles_masked, clim, max_range, time_stamp, method='', show_plot=False):
    # Angle plots
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

class if_time_domain_animation():
    def __init__(self, tfd_angles, s, grid=False):
        t = tfd_angles[0]
        self.fig = plt.figure("IF time domain")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time [s]')
        self.ax.set_ylabel('Voltage [V]')
        self.lines = {}
        for channel in range(1, s['channel_count']+1):  # Channels are 1 based due to hardware considerations
            self.lines[channel] = self.ax.plot(t, np.zeros((len(t),)), label='CH'+str(channel))[0]  # Grab first in list
        self.ax.set_xlim([0, t[-1]])
        self.ax.set_ylim([-1, 1])
        self.ax.legend(loc='best')
        self.ax.grid(grid)

    def update_plot(self, if_data, time_stamp, clim):
        for channel in if_data:
            self.lines[channel].set_ydata(if_data[channel])
        self.ax.set_title('IF time-domain at time T = {:.3f} s'.format(time_stamp))
        # clim is not used but could be used to relay the maximum
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()

    def refresh_data(self, sweep_to_display, s, time_stamp):
        if_data, clim = postprocessing.calculate_if_data(sweep_to_display, s)
        self.update_plot(if_data, time_stamp, 0)

    def __del__(self):
        plt.close(self.fig.number)


class angle_animation():
    def __init__(self, tfd_angles, s, method='angle'):
        d = tfd_angles[2]
        angles = tfd_angles[3]
        angles_masked = angles[tfd_angles[4]]

        self.fig = plt.figure("Angle")
        self.method = method
        fxdb = np.zeros((len(angles_masked), len(d)))

        if self.method == 'polar':
            self.ax = self.fig.add_subplot(111, polar=True)
            self.quad = self.ax.pcolormesh(angles_masked * np.pi / 180, d, fxdb.transpose())
            self.ax.set_xlabel("???")
            self.ax.set_ylabel("???")
        elif self.method == 'cross-range':
            self.ax = self.fig.add_subplot(111)
            #self.ax.set_xlim([0, max_range])
            self.ax.set_xlabel("Range [m]")
            ylim = 90 * np.sin(angles_masked[0] * np.pi / 180)
            ylim = [-ylim, ylim]
            #self.ax.set_ylim(ylim)
            # self.ax.set_ylim([-30, 30])
            self.ax.set_ylabel("Cross-range [m]")
            r, t = np.meshgrid(d, angles_masked * np.pi / 180)
            x = r * np.cos(t)
            y = -r * np.sin(t)
            self.quad = self.ax.pcolormesh(x, y, fxdb)
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

        self.colorbar = self.fig.colorbar(self.quad, ax=self.ax)

    def update_plot(self, fxdb, time_stamp, clim):
        #plt.figure(self.fig.number)  # Grab the focus

        if self.method == 'polar':
            fxdb = fxdb.transpose()
        # https://stackoverflow.com/questions/18797175/animation-with-pcolormesh-routine-in-matplotlib-how-do-i-initialize-the-data
        # https://stackoverflow.com/questions/29009743/using-set-array-with-pyplot-pcolormesh-ruins-figure
        self.quad.set_array(fxdb[:-1,:-1].ravel())
        self.quad.set_clim(clim - 50, clim)  # Updates the colorbar

        self.fig.canvas.flush_events()
        self.fig.canvas.draw()
        self.ax.set_title("Angle plot at T = {:.3f} s".format(time_stamp))

    def __del__(self):
        plt.close(self.fig.number)

def plot_range_time(t, meshgrid_data, m, time_stamp, show_plot=False):
    # Range time plot
    if time_stamp[0]:
        save_path = os.path.join(time_stamp[0], 'Range-time_{:03d}S{}.png'.format(time_stamp[2], time_stamp[3]))
    plt.figure()
    plt.ylabel("Range [m]")
    plt.xlabel("Time [s]")
    plt.xlim([t[0], t[-1]])
    plt.title('Range-time plot')
    imgplot = plt.pcolormesh(*meshgrid_data)
    imgplot.set_clim(m-80,m)
    plt.colorbar()
    if time_stamp[0]:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()

class range_time_animation():
    def __init__(self, s, max_range_index):
        self.fig = plt.figure("Range-time")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim([-s['real_time_recall'], 0])
        self.ax.set_xlabel("Time [s]")
        #self.ax.set_ylim([angles_masked[0], angles_masked[-1]])
        self.ax.set_ylabel("Range [m]")

        # Create the meshgrid
        #t = np.linspace(0, overall_decimate * nb_sweeps * (s['t_sweep'] + s['t_delay']), nb_frames)
        nb_sweeps = int(s['real_time_recall']/s['refresh_period'])+1
        t = np.linspace(-s['real_time_recall'], 0, nb_sweeps, endpoint=True)
        x, y = np.meshgrid(t, np.linspace(0,
        s['c'] * max_range_index * s['if_amplifier_bandwidth'] / (2 * nb_sweeps) / (s['bw'] / s['t_sweep']),
                                          max_range_index-2))

        # Store the value currently displayed for speed
        self.current_array = np.zeros((max_range_index-2, nb_sweeps))
        self.quad = self.ax.pcolormesh(x, y, self.current_array)
        self.colorbar = self.fig.colorbar(self.quad, ax=self.ax)

    def update_plot(self, im, time_stamp, clim):
        # Current values are rolled by the length of the new im array
        self.current_array = np.roll(self.current_array, -1, axis=1)  # Column roll
        self.current_array[:, -1] = im  # Substitute the new array

        self.quad.set_array(self.current_array[:-1, :-1].ravel())  # Flatten the data
        self.quad.set_clim(clim - 80, clim)  # Updates the colorbar
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()
        self.ax.set_title('Range-time plot\nCurrent: {:.3f} | Next: {:.3f}'.format(time_stamp, time_stamp))

    def __del__(self):
        plt.close(self.fig.number)
    """[TBR]
    def display_manager(self, last_sweep):
        # Analyze the last_sweep dictionary
        if last_sweep['counter_sweep'] > self.counter_sweep:
            time_stamp = (None,
                          last_sweep['T'] * last_sweep['counter_sweep'],
                          int(last_sweep['T'] * last_sweep['counter_sweep']),
                          int(1000 * (last_sweep['T'] * last_sweep['counter_sweep'] - int(last_sweep['T'] * last_sweep['counter_sweep']))))
            self.update_plot(last_sweep['last_sweep'], time_stamp, last_sweep['clim'])
            self.counter_sweep = last_sweep['counter_sweep']
    """


