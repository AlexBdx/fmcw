import multiprocessing as mp
from fmcw import adc, ftdi
import time
import os

class fpga_reader(mp.Process):
    """
    This class describes the object that will read the FPGA. By inheriting from Process, it can be launched as a
    subprocess.
    """
    def __init__(self, flag_reading_data, connection, s):
        """
        Nothing too fancy here. Note that initially all the ftdi initialization was done here but the generated object
        cannot be pickled. Therefore, I moved it to the run function.
        :param connection: the end of a Pipe
        :param s: settings dictionary
        """
        mp.Process.__init__(self)  # Calling super constructor - mandatory
        self.connection = connection
        self.s = s

        # Send some basic data back to the parent
        process_info = {
            'pid': os.getpid()
        }
        self.connection.send(process_info)
        self.flag_reading_data = flag_reading_data  # Indicates to everyone that the FPGA is being read.

    def run(self):
        """
        Main routine polling the USB bus. It is not perfect, but >95% of the frames are valid with this configuration.
        :return: void
        """
        self.adf4158 = adc.ADF4158()

        try:
            self.fpga = ftdi.FPGA(self.adf4158, encoding=self.s['encoding'])
        except Exception as e:
            print("[ERROR]", e)
            return

        self.fpga.set_gpio(led=True, adf_ce=True)
        self.fpga.set_adc(oe2=True)
        self.fpga.clear_adc(oe1=True, shdn1=True, shdn2=True)
        real_delay = self.fpga.set_sweep(self.s['f0'], self.s['bw'], self.s['t_sweep'], self.s['t_delay'])  # Returned value delay not used
        print("[INFO] Real delay between sweeps: {} s".format(real_delay))
        self.fpga.set_downsampler(enable=self.s['down_sampler'], quarter=self.s['quarter'])
        self.fpga.write_sweep_timer(self.s['t_sweep'])
        self.fpga.write_sweep_delay(self.s['t_delay'])
        self.fpga.write_decimate(self.s['acquisition_decimate'])
        self.fpga.write_pa_off_timer(self.s['t_delay'] - self.s['pa_off_advance'])
        self.fpga.clear_gpio(pa_off=True)
        self.fpga.clear_buffer()
        self.fpga.set_channels(a=self.s[1], b=self.s[2])  # Would not scale up

        # Run while nothing is put to the pipe - it is an "almost" read only pipe
        self.flag_reading_data.set()  # To maximize accuracy, this is set just before entering the while loop
        while True:
            self.connection.send_bytes(self.fpga.device.read(self.s['byte_usb_read']//self.s['sub_call']))

    def close(self):
        self.flag_reading_data.clear()  # First thing done after the terminate call from parent process
        # Close properly the fpga device
        self.fpga.set_adc(oe1=True, shdn1=True, shdn2=True)
        self.fpga.set_channels(a=False, b=False)
        self.fpga.clear_gpio(led=True, adf_ce=True)
        self.fpga.set_gpio(pa_off=True)
        self.fpga.clear_buffer()
        self.fpga.close()

        print("[INFO] FPGA subprocess successfully closed")
        return