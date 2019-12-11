#!/usr/bin/env python3
"""experiment_viterbi.py
James Gardner 2019
ANU / Melbourne Uni

applies viterbi analysis pipeline to experiment video of
michelson interferometer with a mirror driven at a changing frequency
takes signal as time series of intensity at the video's centre
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm_notebook as tqdm

def fourier_spectrum(signal, fps, return_spectrum=False,
                     produce_plot=False, out_plot_name='tmp.pdf', out_plot_title=''):
    """finds fourier spectrum of signal time series as numpy array,
    has functionality to return and/or plot the sectrum (both default off),
    built from code originally found in tracker_time_series.ipynb
    """   
    signal_frames = len(signal)
    # will drop two frames later, fps: frames per second
    total_time = (signal_frames-2)/fps
    t = np.linspace(0,total_time,signal_frames)
    dt = t[1] - t[0]

    yf = np.fft.fft(signal)
    # normalised-absolute value of FT'd signal
    nrm_abs_yf = 2/signal_frames*np.abs(yf)
    # values at the centre of each frequency bin
    freq_scale = np.fft.fftfreq(len(yf),dt)
    # real signals are symmetric about 0 in frequency domain
    freq_scale_positive = freq_scale[:signal_frames//2]
    # frequency distribution values on positive side
    freq_prob = nrm_abs_yf[:signal_frames//2]
    
    if produce_plot:
        fig, (ax0,ax1) = plt.subplots(2,figsize=(14,14))
        ax0.plot(t,signal)
        ax0.set(title='signal: {}'.format(out_plot_title),ylabel='signal strength',xlabel='time, t')
        # signal average value gives magnitude of frequency = 0 term
        # simple fix is to drop first two bins, otherwise need to shift signal
        ax1.plot(freq_scale_positive[2:],freq_prob[2:])
        ax1.set(title='discrete FFT',ylabel='freq strength in signal',xlabel='frequency, f')
        plt.savefig(out_plot_name,bbox_inches='tight')
        plt.clf()
    
    if return_spectrum:
        return freq_prob[2:], freq_scale_positive[2:]
    
def series_at_point(filename, point=None, return_series=False,
                    produce_plot=False, out_plot_name='tmp.pdf'):
    """finds time series green-channel intensity at a point in the video,
    has functionality to both return and plot series (both default off)
    """
    # standard python3-openCV point capture
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames-1)

    point_intensity = []
    # frame is numpy array of b,g,r values at each pixel
    ret, frame = cap.read()
    if point is None:
        # default to centre of frame
        point = tuple([int(i/2) for i in frame.shape[:2]])

    while ret:
        # green channel is the most important for greyscale intensity
        # https://en.wikipedia.org/wiki/Relative_luminance        
        # so approximate greyscale as just the green channel
        point_intensity.append(frame[point][1])

        ret, frame = cap.read()
        pbar.update(1)

    pbar.close()
    cap.release()

    if produce_plot:
        fourier_spectrum(point_intensity, fps, produce_plot=True,
                         out_plot_name=out_plot_name, out_plot_title=filename)
    if return_series:
        return point_intensity, fps    

class PointViterbi(object):
    """finds viterbi path through frequency spectrum measured at
    centre point over time for changing driving frequency
    """      
    def __init__(self,filename,long_timesteps=None,scanning_range=3):
        # start by finding the signal to be split into bins
        long_signal, fps = series_at_point(filename,return_series=True)
        
        total_frames = len(long_signal)
        duration = total_frames/fps
        if long_timesteps is None:
            # for every minute, add an extra 20 long time bins
            long_timesteps = int(20*duration/60)
        # q,r = divmod(a,b) s.t. a = q*b+r
        bin_frames, bin_remainder = divmod(total_frames,long_timesteps)
        # bin_duration = bin_frames/fps
        # acts as flag to stop short of the remainder, which is lost
        bin_last = total_frames - bin_remainder
        # always has long_timesteps number of chunks
        bin_signals = [long_signal[i: i+bin_frames] for i in range(0, bin_last, bin_frames)]

        # creating the signal grid
        # positive side of the fourier spectrum will have half bin_frames
        # minus 2 from cutting out the average value, frequency = 0, signal
        grid_frames = bin_frames//2-2
        grid = np.zeros((grid_frames,long_timesteps))

        for i,signal in enumerate(bin_signals):
            # columns are each spectrum, rows are frequency through time
            col, freq_scale_cut = fourier_spectrum(signal, fps, return_spectrum=True)
            grid[:,i] = col

        # normalised grid, algorithm goal is to maximise product of values
        ngrid  = grid/np.max(grid)
        # logarithm avoids underflow, equvivalent to maximise sum of log of values
        lngrid = np.log(ngrid)

        # keep track of running scores for best path to each node
        score_grid  = np.copy(lngrid)
        pathfinder_flag = len(lngrid[:,0])
        # pathfinder stores the survivor paths, i.e. the previous best step
        # to allow back-tracking to recover the best total path at the end
        pathfinder = np.full(np.shape(lngrid), pathfinder_flag)
        # pathfinder flag+1 for reaching the first, 0-index column        
        pathfinder[:,0] = pathfinder_flag+1       

        # implementation of the viterbi algorithm itself
        # finding the best path to each node, through time
        # see: https://www.youtube.com/watch?v=6JVqutwtzmo
        for j in range(1,long_timesteps):
            for i in range(len(score_grid[:,j])):
                # index values for where to look relative to i in previous column
                k_a = max(0, i-scanning_range) 
                k_b = min(len(score_grid[:,j-1])-1,
                          i+scanning_range)
                window = score_grid[:,j-1][k_a:k_b+1]
                # find the best thing nearby in the previous column ...
                window_score = np.max(window)
                window_ref   = k_a+np.argmax(window)
                # ... and take note of it, summing the log of values
                score_grid[i][j] += window_score
                pathfinder[i][j] = window_ref 

        # look at the very last column, and find the best total ending ...
        best_score  = np.max(score_grid[:,-1])
        best_end = np.argmax(score_grid[:,-1])
        # ... and retrace its steps through the grid
        best_path_back = np.full(long_timesteps,pathfinder_flag+2)
        best_path_back[-1] = best_end
        # best_path_back is the viterbi path, the highest scoring overall 
        # path_grid is the binary image of the viterbi path taken
        path_grid = np.zeros(np.shape(ngrid))
        tmp_path = pathfinder[best_end][-1]

        for j in reversed(range(0,long_timesteps-1)):
            path_grid[tmp_path][j] = 1
            # take pathfinder value in current step and follow it backwards
            best_path_back[j] = tmp_path    
            tmp_path = pathfinder[tmp_path][j]

        # make sure we got all the way home
        # (that the retrace found the initial edge)
        assert tmp_path == pathfinder_flag+1
        
        self.ngrid = ngrid
        self.path_grid = path_grid
        # _plot_bundle not meant to be accessed other than for plotting
        self._plot_bundle = filename, long_timesteps, duration, grid_frames, freq_scale_cut 

    def plot(self, filetag='tmp'):
        """saves plots of the signal grid and the recovered viterbi path,
        filetag is added to standardised plot filenames
        """
        # mirror _plot_bundle assignment above
        filename, long_timesteps, duration, grid_frames, freq_scale_cut = self._plot_bundle
        
        # adaptive tick marks to video duration
        tick_skip = int(duration//100)
        xtick_labels_0 = (np.linspace(0,1,long_timesteps)*duration).astype(int)
        xtick_labels_1 = []
        for i in range(0,long_timesteps,tick_skip):
            xtick_labels_1.append(xtick_labels_0[i])
            for _ in range(tick_skip):
                xtick_labels_1.append('')
        ytick_labels_0 = ['{:.2f}'.format(i) for i in freq_scale_cut]
        ytick_labels_1 = []
        for i in range(0,grid_frames,tick_skip):
            ytick_labels_1.append(ytick_labels_0[i])
            for _ in range(tick_skip):
                ytick_labels_1.append('')          
        
        plt.figure(figsize=(7,14))
        plt.imshow(self.ngrid, cmap='viridis');
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')  
        plt.xticks(np.arange(long_timesteps),xtick_labels_1, rotation=90)
        plt.yticks(np.arange(grid_frames),ytick_labels_1)
        cbar_fraction = 0.025
        cbar = plt.colorbar(fraction=cbar_fraction) 
        cbar.set_label('normalised frequency distribution')
        plt.title('{}\n fourier spectrum of signal binned over time\n'.format(filename))
        plt.ylabel('signal frequency, f / Hz')
        plt.xlabel('long time duration, t / s')
        plt.savefig('expt_ngrid_{}.pdf'.format(filetag),bbox_inches='tight')
        plt.clf()
        
        plt.figure(figsize=(7,14))
        plt.imshow(self.path_grid, cmap='viridis');
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')  
        plt.xticks(np.arange(long_timesteps),xtick_labels_1, rotation=90)
        plt.yticks(np.arange(grid_frames),ytick_labels_1)
        plt.title('{}\n viterbi path through signal grid'.format(filename))
        plt.ylabel('signal frequency, f / Hz')
        plt.xlabel('long time duration, t / s')
        plt.savefig('expt_viterbi_path_{}.pdf'.format(filetag),bbox_inches='tight')
        plt.clf()
        
if __name__ == '__main__':
    # apply viterbi analysis pipeline to roughly 15 minutes of experiment video 
    PointViterbi('expt_5_fast.mp4').plot('5_fast')
    PointViterbi('expt_5_slow.mp4').plot('5_slow')
    PointViterbi('expt_5_high.mp4').plot('5_high')
    PointViterbi('expt_5_long.mp4').plot('5_long')    