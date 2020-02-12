#!/usr/bin/python
"""raspberry_pi_photodiode_adc.py
James Gardner 2020
built from existing scripts online since lost,
gratitude to those original authors

to be run on a Rasperry Pi,
reads the signal from an ADC and saves to a .csv file
"""

import spidev
import time
import numpy as np
import matplotlib.pyplot as plt

# define sensor channels
light_channel = 0
# define delay between readings
delay = 0.2

# open SPI bus
spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz=1000000

def read_channel(channel):
    """returns a single value from a channel of the ADC,
    takes SPI data from MCP3008 chip,
    channel must be an integer 0-7"""
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

def convert_volts(data, places):
    """function to convert data to voltage level,
    rounded to specified number of decimal places"""
    volts = (3.3*data) / float(1023)
    volts = round(volts, places)
    return volts

def live_plot():
    """displays a live plot of the channel reading,
    can get very slow if left for too long"""
    delay = 1e-7
    t0 = time.time()
    x, y = [], []

    while True:
        t1 = time.time()
        x.append(t1-t0)
        y.append(read_channel(light_channel))
        plt.plot(x, y, 'k')
        plt.pause(delay)

    plt.show()

def save_stream(duration=1, out_file='adc_time_series.csv', show_plot=False):
    """saves a data stream from the ADC for the given duration to a .csv files,
    sample rate is just above 16kHz,
    show_plot is not recommend due to the large number of points"""
    with open(out_file, 'w') as f:
        t0 = time.time()
        while True:
            t1 = time.time()
            t = t1-t0
            light_level = read_channel(light_channel)
            f.write('{},{}\n'.format(t,light_level))
            if t > duration:
                break

    if show_plot:
        time_series = np.genfromtxt(out_file,delimiter=',')
        print('resolution = {:.3f}Hz'.format(len(time_series)/duration))
        plt.plot(time_series[:,0], time_series[:,1],'-k')
        plt.show()

if __name__ == '__main__':
    print('Welcome to raspberry_pi_photodiode_adc.py, attempting to save a one second stream now')
    try:
        save_stream(duration=1)
    except:
        print('This script must be run on a Raspberry Pi that is set-up \
              exactly as in the GravExplain paper, please try again')
