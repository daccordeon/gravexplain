#!/usr/bin/env python3
"""mock_continuous_signal.ipynb
James Gardner 2019
ANU / Melbourne Uni

injects a noisy sinusoidal signal with varying frequency over time (in discrete bins),
applies (discrete fast) fourier transform to each individual sinusoid,
then applies viterbi's algorithm through time and recovers the meandering frequency
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from tqdm import tqdm_notebook as tqdm


class SineSignal(object):
    """creates a noisy sinusoidal signal with frequency of raw_freq,
    signal over n_t points between 0 and 2*pi with signal:noise of noise_scale,
    then applies discrete FFT and stores the normalised frequency distribution as freq_prob
    """
    n_t = int(1e3)
    # ultimately, freq_prob will be half this size
    t = np.linspace(0,2*np.pi,n_t)    
    dt = t[1]-t[0]
    # e.g. noise_scale = 6 if signal:noise = 1:6
    noise_scale = 6
    
    def __init__(self,raw_freq):
        self.raw_freq = raw_freq
        self.raw_signal = np.sin(raw_freq*2*np.pi*SineSignal.t)
        self.noise = np.random.normal(0,SineSignal.noise_scale,self.raw_signal.shape)
        self.injected_signal = self.raw_signal + self.noise
        
        self.yf = np.fft.fft(self.injected_signal)
        # normalised-absolute value of FT'd injected signal
        self.nrm_abs_yf = 2/SineSignal.n_t*np.abs(self.yf)
        self.inj_freq = np.fft.fftfreq(len(self.yf),SineSignal.dt)
        # np.fft.fftfreq outputs 0 to +inf then -inf to 0, so :N//2 gets +ve side; wild!
        self.freq_prob = self.nrm_abs_yf[:SineSignal.n_t//2]
        
    def plot(self):
        """saves plots of injected signal in time and frequency domains"""
        fig, (ax0,ax1) = plt.subplots(2,figsize=(14,14))
        ax0.plot(SineSignal.t,self.injected_signal)
        ax1.plot(self.inj_freq[:SineSignal.n_t//2],self.freq_prob)
        ax0.set(title='injected signal',ylabel='signal value',xlabel='signal time chunk')
        ax1.set(title='discrete FFT',ylabel='freq strength in signal',xlabel='frequency')
        plt.savefig('example_signal.pdf',bbox_inches='tight')
        plt.clf()
        
        
class SignalGrid(object):
    """stitches frequency distribution of sinusoidal signals into a
    grid with the frequency slowly changing (meandering) through time,
    then applies viterbi's algorithm to try to recover the frequency path"""
    # long time as opposed to the short 0 to 2*pi interval of each signal
    long_timesteps = 100
    bin_time = np.linspace(0,1,long_timesteps)
    # meander is the long scale change in the sine frequency
    meander_amp = 20
    meander_decay = 2
    meander_freq = 2
    meander = lambda x: SignalGrid.meander_amp*(
        np.exp(-x*SignalGrid.meander_decay)*
        np.sin(SignalGrid.meander_freq*2*np.pi*x))
    # connections back a timestep made for all indicies with plus/minus scanning_range
    # this limits how much the viterbi path will change frequency at each timestep made
    scanning_range = 10
    initial_frequency = 20
    
    def __init__(self):  
        # halve total number of points due to fft on real function
        self.grid = np.zeros((SineSignal.n_t//2,SignalGrid.long_timesteps))
        # print('grid is:',grid.shape)
        # signal meanders from the initial frequency
        self.wandering_freqs = (SignalGrid.initial_frequency+
                                SignalGrid.meander(SignalGrid.bin_time))
        # post_freq is the maximum frequency in the column
        # if the resultant plot is exact, then the recovery is perfect
        self.post_freq = []

        for i,f in tqdm(enumerate(self.wandering_freqs)):
            thing_f = SineSignal(f)
            col = thing_f.freq_prob
            # [i][j] same as [:,j][i] same as [i,j]
            self.grid[:,i] = col
            self.post_freq.append(thing_f.inj_freq[col.argmax()])

        # normalised grid, trying to maximise produce of values
        self.ngrid  = self.grid/np.max(self.grid)
        # logarithm avoids underflow, maximise sum of log(value)'s
        self.lngrid = np.log(self.ngrid)
        
        self.score_grid  = np.copy(self.lngrid)
        self.pathfinder_flag = len(self.lngrid[:,0]) #=500
        # pathfinder stores the survivor paths, to allow back-tracking through
        self.pathfinder = np.full(np.shape(self.lngrid), self.pathfinder_flag)
        # pathfinder flag+1 for reaching the first, 0-index column        
        self.pathfinder[:,0] = self.pathfinder_flag+1       

        # the viterbi algorithm, through time finding the best path to each node
        # see: https://www.youtube.com/watch?v=6JVqutwtzmo
        for j in tqdm(range(1,SignalGrid.long_timesteps)): #range(100)
            for i in range(len(self.score_grid[:,j])): #range(500)
                # index values for where to look relative to i in previous column
                k_a = max(0, i-SignalGrid.scanning_range) 
                k_b = min(len(self.score_grid[:,j-1])-1,
                          i+SignalGrid.scanning_range)
                #print(k_a,k_b)
                window = self.score_grid[:,j-1][k_a:k_b+1]
                # find the best thing nearby in the previous column ...
                window_score = np.max(window)
                window_ref   = k_a+np.argmax(window)
                # ... and take note of it, summing the log(value)'s
                self.score_grid[i][j] += window_score
                self.pathfinder[i][j] = window_ref 

        # look at the very last column, and find the best ending for the path
        best_score  = np.max(self.score_grid[:,-1])
        best_end = np.argmax(self.score_grid[:,-1])
        # now need to retrace the steps through the grid
        self.best_path_back = np.full(SignalGrid.long_timesteps,self.pathfinder_flag+2)
        self.best_path_back[-1] = best_end

        # path_grid is the binary image of the viterbi path taken
        self.path_grid = np.zeros(np.shape(self.ngrid))
        tmp_path = self.pathfinder[best_end][-1]

        for j in tqdm(reversed(range(0,SignalGrid.long_timesteps-1))):
            self.path_grid[tmp_path][j] = 1
            # take pathfinder value in current step and follow it backwards
            self.best_path_back[j] = tmp_path    
            tmp_path = self.pathfinder[tmp_path][j]

        # make sure we got all the way home
        assert tmp_path == self.pathfinder_flag+1

    def plot(self):
        """saves plots of meandering frequency, the signal grid, and the recovered viterbi path """
        plt.figure(figsize=(14,7))
        plt.plot(SignalGrid.bin_time,
                 SignalGrid.meander(SignalGrid.bin_time)+
                 SignalGrid.initial_frequency,'.',label='meandering frequency')    
        plt.plot(SignalGrid.bin_time,self.post_freq,'r.',label='highest strength frequency')
        plt.legend()
        plt.title('highest strength frequency and actual meandering frequency of signal')
        plt.ylabel('signal frequency')
        plt.xlabel('bin time')
        plt.savefig('meandering_frequency.pdf',bbox_inches='tight')
        plt.clf()

        plt.figure(figsize=(7,14))
        plt.imshow(self.ngrid, cmap='viridis')
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')
        cbar = plt.colorbar() 
        cbar.set_label('frequency probability distribution')
        plt.title('grid of signals in frequency domain as frequency changes')
        plt.ylabel('signal frequency bins')
        plt.xlabel('long time bins')
        plt.savefig('signal_grid_raw.pdf',bbox_inches='tight')
        plt.clf()

        plt.figure(figsize=(7,14))
        plt.imshow(self.lngrid, cmap='viridis')
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')        
        cbar = plt.colorbar() 
        cbar.set_label('log(frequency) probability distribution')
        plt.title('grid of signals in log(frequency) domain as frequency changes')
        plt.ylabel('signal frequency bins')
        plt.xlabel('long time bins')
        plt.savefig('lnwandering.pdf',bbox_inches='tight')
        plt.clf()

        plt.figure(figsize=(7,14))
        plt.imshow(self.path_grid, cmap='viridis')
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')        
        plt.title('viterbi path through signal frequency grid')
        plt.ylabel('signal frequency bins')
        plt.xlabel('long time bins')
        plt.savefig('viterbi_path.pdf',bbox_inches='tight')
        plt.clf()
        

if __name__ == '__main__':
    SineSignal(raw_freq=5).plot()
    SignalGrid().plot()
