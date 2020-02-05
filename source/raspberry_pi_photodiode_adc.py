#!/usr/bin/python
"""raspberry_pi_photodiode_adc.py
James Gardner 2020

reads ADC signal and saves to .csv
"""

import spidev
import time
#import os
import numpy as np
import matplotlib.pyplot as plt

# Define sensor channels
light_channel = 0
# Define delay between readings
delay = 0.2

# Open SPI bus
spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz=1000000

# Function to read SPI data from MCP3008 chip
# Channel must be an integer 0-7
def read_channel(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

# Function to convert data to voltage level,
# rounded to specified number of decimal places.
def convert_volts(data, places):
    volts = (3.3*data) / float(1023)
    volts = round(volts, places)
    return volts

def live_plot():
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
    save_stream(duration=1)
