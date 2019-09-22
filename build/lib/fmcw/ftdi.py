import pylibftdi as ftdi
from queue import Queue, Empty
from threading import Thread
import datetime

class FPGA():
    """
    Creates the FTDI object that handles the communication with the FPGA.
    """
    def __init__(self, ADC, encoding='latin1'):
        """Set up the FTDI object that represents the FPGA.

        :param ADC: ADC object (see ADC module)
        :param encoding: Encoding of the text data coming from the FTDI object
        """
        SYNCFF = 0x40
        SIO_RTS_CTS_HS = (0x1 << 8)
        #self.device = ftdi.Device(mode='t', interface_select=ftdi.INTERFACE_A, encoding=encoding)
        self.device = ftdi.Device(mode='b', interface_select=ftdi.INTERFACE_A, encoding=encoding)
        #self.device.open()  # Not needed if not lazy open
        self.device.ftdi_fn.ftdi_set_bitmode(0xff, SYNCFF)
        self.device.ftdi_fn.ftdi_read_data_set_chunksize(0x10000)
        self.device.ftdi_fn.ftdi_write_data_set_chunksize(0x10000)
        self.device.ftdi_fn.ftdi_setflowctrl(SIO_RTS_CTS_HS)
        self.device.flush()
        print("[INFO] FTDI baudrate:", self.device.baudrate)

        self.pll = ADC  # ADC and PLL are on the same clock
        self.fclk = 40e6  # [Hz] Clock frequency
        self.fpd_freq = self.fclk/2

    def close(self):
        """
        Close the FTDI object
        :return:
        """
        self.device.close()
        return

    def send_packet(self, x, cmd):
        """
        Add a header to a packet, encode it and write to FTDI.
        :param x: data
        :param cmd: type of packet
        :return: write to FTDI
        """
        try:
            l = len(x)
        except TypeError:
            l = 1
            x = [x]
        
        header = [0xaa, l, cmd]
        b = bytearray(header+x)
        #print map(hex, map(ord, str(b)))
        return self.device.write(b)
        

    def clear_gpio(self, led=False, pa_off=False, mix_enbl=False, adf_ce=False):
        """
        Create a packet signaling the GPIO pins to clear on the FPGA.
        :param led: Clear the LED
        :param pa_off: Clear pa_off
        :param mix_enbl: Disable mixer
        :param adf_ce: Clear the adf_ce
        :return: Packet to be encapsulated
        """
        w = 0
        if led:
            w |= 1 << 0
        if pa_off:
            w |= 1 << 1
        if mix_enbl:
            w |= 1 << 2
        if adf_ce:
            w |= 1 << 3
        return self.send_packet(w, 0)

    def set_gpio(self, led=False, pa_off=False, mix_enbl=False, adf_ce=False):
        """
        Create a packet signaling the GPIO pins to set on the FPGA.
        :param led: Set the LED
        :param pa_off: Set the pa_off
        :param mix_enbl: Enable the mixer
        :param adf_ce: Set the adf_ce
        :return: Packet to be encapsulated
        """
        w = 0
        if led:
            w |= 1 << 0
        if pa_off:
            w |= 1 << 1
        if mix_enbl:
            w |= 1 << 2
        if adf_ce:
            w |= 1 << 3
        return self.send_packet(w, 1)

    def clear_adc(self, oe1=False, oe2=False, shdn1=False, shdn2=False):
        """
        Create a packet signaling the ADC pins to clear on the FPGA.
        :param oe1: Clear Output Enable 1
        :param oe2: Clear Output Enable 2
        :param shdn1: Clear Shutdown 1
        :param shdn2: Clear Shutdown 2
        :return: Packet to be encapsulated
        """
        w = 0
        if oe1:
            w |= 1 << 0
        if oe2:
            w |= 1 << 1
        if shdn1:
            w |= 1 << 2
        if shdn2:
            w |= 1 << 3
        return self.send_packet(w, 2)

    def set_adc(self, oe1=False, oe2=False, shdn1=False, shdn2=False):
        """
        Create a packet signaling the ADC pins to set on the FPGA.
        :param oe1: Set Output Enable 1
        :param oe2: Set Output Enable 2
        :param shdn1: Set Shutdown 1
        :param shdn2: Set Shutdown 2
        :return: Packet to be encapsulated
        """
        w = 0
        if oe1:
            w |= 1 << 0
        if oe2:
            w |= 1 << 1
        if shdn1:
            w |= 1 << 2
        if shdn2:
            w |= 1 << 3
        return self.send_packet(w, 3)

    def write_pll_reg(self, n):
        """
        Create a packet to configure a PLL register.
        :param n: Configuration parameter
        :return: Packet to be encapsulated
        """
        reg = self.pll.registers[n]
        w = [(reg & 0xff) | n, (reg >> 8) & 0xff, (reg >> 16) & 0xff, (reg >> 24) & 0xff]
        return self.send_packet(w, 4)

    def write_pll(self):
        """
        Call for the configuration of all the registers.
        :return: void
        """
        self.write_pll_reg(7)
        self.pll.write_value(step_sel=0)
        self.write_pll_reg(6)
        self.pll.write_value(step_sel=1)
        self.write_pll_reg(6)
        self.pll.write_value(dev_sel=0)
        self.write_pll_reg(5)
        self.pll.write_value(dev_sel=1)
        self.write_pll_reg(5)
        for i in range(4,-1,-1):
            self.write_pll_reg(i)

    def set_sweep(self, fstart, bw, length, delay):
        """
        Set sweep parameters.
        :param fstart: [Hz] Start frequency of the chirp
        :param bw: [Hz] Bandwidth to use
        :param length: [s] Duration of the sweep
        :param delay: [s] Delay between two sweeps
        :return: real delay between two sweeps (just informational)
        """
        real_delay = self.pll.freq_to_regs(fstart, self.fpd_freq, bw, length, delay)  # Configure the PLL
        self.write_pll()
        return real_delay

    def write_sweep_timer(self, length):
        """
        Convert the duration of the sweep to a number of ADC clock cycles
        :param length: number of clock cycles that represent the duration of a sweep
        :return: Packet to be encapsulated
        """
        length = int(self.fclk*length)
        w = [(length & 0xff), (length >> 8) & 0xff, (length >> 16) & 0xff, (length >> 24) & 0xff]
        return self.send_packet(w, 5)

    def write_sweep_delay(self, length):
        """
        Convert the duration between two sweeps (sweep delay) to a number of ADC clock cycles
        :param length: number of clock cycles that represent the duration between two sweeps
        :return: Packet to be encapsulated
        """
        length = int(self.fclk*length)
        w = [(length & 0xff), (length >> 8) & 0xff, (length >> 16) & 0xff, (length >> 24) & 0xff]
        return self.send_packet(w, 6)

    def set_channels(self, a=True, b=True):
        """WARNING: ONLY 2 CHANNELS SUPPORTED
        Set the channels to be activated (only two supported)
        :param a: State of channel a
        :param b: State of channel b
        :return: Packet to be encapsulated
        """
        w = 0
        if a:
            w |= 1 << 0
        if b:
            w |= 1 << 1
        return self.send_packet(w, 7)

    def set_downsampler(self, enable=True, quarter=False):
        """WARNING: THE FPGA CODE REQUIRES IT TO BE ENABLED. TO DO: ALLOW THE USER TO DEACTIVATE IT
        Set the downsampler.
        :param enable: Turn it on
        :param quarter: Divide the sampling rate by another factor of 2
        :return: Packet to be encapsulated
        """
        w = 0
        if enable:
            w |= 1 << 0
        if quarter:
            w |= 1 << 1
        return self.send_packet(w, 8)

    def write_decimate(self, decimate):
        """
        Create a packet to configure the decimation factor at the FPGA level.
        :param decimate: Number of sweeps to skip. 0 means no sweeps are skipped.
        :return: Packet to be encapsulated
        """
        """[Old Henrik] Used to be 1 based
        if decimate > 2**16-1 or decimate < 1:
            raise ValueError("Invalid decimate value")
        decimate = int(decimate) - 1
        """
        # Sanity checks
        if type(decimate) != int:
            raise TypeError('decimate needs to be an int')
        if decimate > 2**16-1 or decimate < 0:
            raise ValueError("Invalid decimate value: {} is not between {} and {}".format(decimate, 0, 2**16-1))
        
        w = [decimate & 0xff, (decimate >> 8) & 0xff]
        return self.send_packet(w, 9)

    def write_pa_off_timer(self, length):
        """
        Convert the duration pa_off_timer to a number of ADC clock cycles
        :param length: number of clock cycles that represent the pa_off_timer
        :return: Packet to be encapsulated
        """
        length = int(self.fclk*length)
        w = [(length & 0xff), (length >> 8) & 0xff, (length >> 16) & 0xff, (length >> 24) & 0xff]
        return self.send_packet(w, 10)

    def clear_buffer(self):
        """
        Clear some buffer
        :return: Packet to be encapsulated
        """
        return self.send_packet(0, 11)


class Writer(Thread):
    """DEPRECATED
    Legacy Writer thread used to write to the binary log file
    """
    def __init__(self, filename, queue, encoding='latin1', timeout=0.5):
        Thread.__init__(self)
        
        self.queue = queue
        self.f = open(filename, 'w', encoding=encoding)
        self.timeout = timeout
        print("[INFO] Opened {} for writing".format(filename))

    def run(self):
        wrote = 0
        freq = 1
        time_tracker = 0
        while True:
            try:
                d = self.queue.get(True, self.timeout)
            except Empty:
                print('[ERROR] Timeout after {} s without data | Wrote {} byte'.format(self.timeout, wrote))
                self.f.close()
                return
            if len(d) == 0:
                print('\n[INFO] Done after writing {:,} byte'.format(wrote))
                self.f.close()
                return
            else:
                self.f.write(d)
                #self.f.write("\n")
                
                wrote += len(d)
                #print("[INFO] Written {:,} byte to file".format(wrote))
