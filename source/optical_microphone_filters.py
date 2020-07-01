#!/usr/bin/env python3
"""experiment_viterbi.py
James Gardner 2020
ANU / Melbourne Uni

analysis of data from photodiode reading the pattern of
the optical microphone interferometer
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io.wavfile as wavfile
import scipy.signal as ssignal
from logmmse import logmmse

def fourier_spectrum_2(signal, fps, return_spectrum=False, cutter=2, remove_mains=False,
                     produce_plot=False, out_plot_name='tmp.pdf', out_plot_title=''):
    """finds fourier spectrum of signal time series as numpy array,
    has functionality to return and/or plot the sectrum (both default off),
    built from code originally found in tracker_time_series.ipynb
    """   
    signal_frames = len(signal)
    
    # will drop two frames later, fps: frames per second
    total_time = (signal_frames-cutter)/fps
    t = np.linspace(0,total_time,signal_frames)
    dt = t[1] - t[0]

    yf = np.fft.fft(signal)
    # normalised-absolute value of FT'd signal
    nrm_abs_yf = 2/signal_frames*np.abs(yf)
    # values at the centre of each frequency bin
    freq_scale = np.fft.fftfreq(len(yf),dt)
    # real signals are symmetric about 0 in frequency domain
    freq_scale_positive = freq_scale[cutter:signal_frames//2]
    # frequency distribution values on positive side
    freq_prob = nrm_abs_yf[cutter:signal_frames//2]
    
    # freqscale 0, +inf, -inf, -0
    #print(freq_scale[:10], freq_prob[:10])#,freq_scale[-5])
    
    if produce_plot:
        
        # tune out 50Hz mains noise
        # closest value in scale is index 5
        #to_go = None#(5,)#tuple(range(10))#(5,)
        # mains_mark
        to_go = ()
        
        if remove_mains:
            mm_0, mm_1 = None, None
            for i, f in enumerate(freq_scale_positive):
                if f > 50:
                    mm_0, mm_1 = i-1, i
                    break
            mm_f_0 = freq_scale_positive[mm_0]
            mm_f_1 = freq_scale_positive[mm_1]
            if abs(mm_f_0-50) < abs(mm_f_1-50):
                to_go = (mm_0,)
            else:
                to_go = (mm_1,)
            
        #print(to_go)
        to_go = tuple(to_go[0]+i for i in (-1,0,1))
        
        freq_prob_mained = np.delete(freq_prob, to_go)
        freq_scale_positive_mained = np.delete(freq_scale_positive, to_go)        
        
        #print(np.abs(yf)[:7])
        to_go_pandn = to_go + tuple(len(yf)-i for i in to_go)
        #print(to_go_pandn)
        
        yf_mained = np.delete(yf, to_go_pandn)
        t_mained = np.delete(t, to_go_pandn)
        iyfm = np.fft.ifft(yf_mained)
        
        fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(21,14))
        ax0,ax1,ax2,ax3 = axes[0,0],axes[1,0],axes[0,1],axes[1,1]
        ax0.plot(t, signal)
        ax0.set(title='signal: {}'.format(out_plot_title),ylabel='signal strength',xlabel='time, t')
        ax2.plot(t_mained,iyfm)
        ax2.set(title='i mained signal: {}'.format(out_plot_title),ylabel='signal strength',xlabel='time, t')
        # signal average value gives magnitude of frequency = 0 term
        # simple fix is to drop first two bins, otherwise need to shift signal
        ax1.plot(freq_scale_positive,freq_prob)
        ax1.set(title='discrete FFT',ylabel='freq strength in signal',xlabel='frequency, f')        
        ax3.plot(freq_scale_positive_mained,freq_prob_mained)
        ax3.set(title='mained discrete FFT',ylabel='freq strength in signal',xlabel='frequency, f')
        plt.savefig(out_plot_name,bbox_inches='tight')
        plt.close(fig)
    
    if return_spectrum:
        return freq_prob, freq_scale_positive  
    
def cut_mains(signal, fps, cutter=2):
    """returns signal with the 50Hz mains cut out
    NB: this is a bad signal processing technique and should not be used!
    """
    signal_frames = len(signal)
    
    total_time = (signal_frames-cutter)/fps
    t = np.linspace(0,total_time,signal_frames)
    dt = t[1] - t[0]

    yf = np.fft.fft(signal)
    # normalised-absolute value of FT'd signal
    nrm_abs_yf = 2/signal_frames*np.abs(yf)
    # values at the centre of each frequency bin
    freq_scale = np.fft.fftfreq(len(yf),dt)
    # real signals are symmetric about 0 in frequency domain
    freq_scale_positive = freq_scale[cutter:signal_frames//2]
    # frequency distribution values on positive side
    freq_prob = nrm_abs_yf[cutter:signal_frames//2]

    # tune out 50Hz mains noise
    to_go = ()
    mm_0, mm_1 = None, None
    for i, f in enumerate(freq_scale_positive):
        if f > 50:
            mm_0, mm_1 = i-1, i
            break
    mm_f_0 = freq_scale_positive[mm_0]
    mm_f_1 = freq_scale_positive[mm_1]
    if abs(mm_f_0-50) < abs(mm_f_1-50):
        to_go = (mm_0,)
    else:
        to_go = (mm_1,)

    #print(to_go)
    #to_go = tuple(to_go[0]+i for i in (-1,0,1))

    freq_prob_mained = np.delete(freq_prob, to_go)
    freq_scale_positive_mained = np.delete(freq_scale_positive, to_go)        

    #print(np.abs(yf)[:7])
    to_go_pandn = to_go + tuple(len(yf)-i for i in to_go)
    #print(to_go_pandn)

    yf_mained = np.delete(yf, to_go_pandn)
    t_mained = np.delete(t, to_go_pandn)
    iyfm = np.fft.ifft(yf_mained)

    return np.abs(iyfm), t_mained

def viterbi_pathfinder(grid, scanning_range=3):
    """find the highest scoring path through the grid, left-to-right,
    as by the viterbi algorithm, with connections plus-minus the scanning_range;
    returns score grid for best path to each node and a bitmap of the total best path
    """
    # normalised grid, algorithm goal is to maximise product of values
    ngrid  = grid/np.max(grid)
    # logarithm avoids underflow, equvivalent to maximise sum of log of values
    lngrid = np.log(ngrid)
    
    long_timesteps = grid.shape[1]

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
    
    return score_grid, path_grid

def absmax(axis):
    """returns the maximum absolute value along an axis of a np.array"""
    return max(abs(np.min(axis)), abs(np.max(axis)))

def photodiode_experiment_viterbi(filename='podo_viterbi_test.csv',
                                  filetag='viterbi_test'):
    """performs viterbi analysis on window of .wav recording from photodiode"""
    # read in photodiode (aka podo) data as time series
    mega = np.genfromtxt(filename,delimiter=',')
    times = mega[:,0]
    long_signal = mega[:,1]
    fps = len(mega)/mega[-1,0]

    total_frames = len(long_signal)
    duration = total_frames/fps
    print('duration: {:.1f}s, fps: {:.1f}/s, frames: {}'.format(duration, fps, total_frames))

    # save an initial audio recording of the response
    wavfile.write('podo_{}_raw.wav'.format(filetag), int(fps),
                  long_signal.astype('float32')/absmax(long_signal))

    # processing a window of the time series
    window_start = 5
    window_duration = 0.5
    window_start_frame = int(window_start/duration*total_frames)
    window_size = int(window_duration/duration*total_frames)

    window_signal = long_signal[window_start_frame:window_start_frame+window_size]
    window_times = times[window_start_frame:window_start_frame+window_size]
    window_prob, window_fscale = fourier_spectrum_2(window_signal, fps, return_spectrum=True)

    fig, (ax0,ax1) = plt.subplots(2,figsize=(14,7))
    ax0.plot(window_times, window_signal)
    ax0.set(title='window signal',ylabel='photodiode reading',xlabel='time, t / s')
    ax1.plot(window_fscale, window_prob)
    ax1.set(ylabel='fourier strength',xlabel='frequency, f / Hz')
    plt.savefig('podo_{}_window.pdf'.format(filetag))
    plt.close(fig)

    # apply viterbi analysis to long_signal
    long_timesteps = 600
    scanning_range = 3

    # q,r = divmod(a,b) s.t. a = q*b+r
    bin_frames, bin_remainder = divmod(total_frames,long_timesteps)
    # bin_duration = bin_frames/fps
    # acts as flag to stop short of the remainder, which is lost
    bin_last = total_frames - bin_remainder
    # always has long_timesteps number of chunks
    bin_signals = [long_signal[i: i+bin_frames] for i in range(0, bin_last, bin_frames)]

    # creating the signal grid
    grid_frames = bin_frames//2
    grid = np.zeros((grid_frames,long_timesteps))

    for i, signal in enumerate(bin_signals):
        # columns are each spectrum, rows are frequency through time
        col, freq_scale_cut = fourier_spectrum_2(signal, fps, return_spectrum=True,cutter=0)
        grid[:,i] = col

    # fgrid ignores the first 100 bins, to avoid mains noise
    fgrid = grid[100:,:]
    ngrid = fgrid/np.max(fgrid)    

    path_grid = viterbi_pathfinder(fgrid, scanning_range)[1]
    print('viterbi path found')

    # save plots of grid and viterbi path through it

    plt.figure(figsize=(7,14))
    plt.imshow(ngrid, cmap='viridis');
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')  
    # plt.xticks(np.arange(long_timesteps),xtick_labels_1, rotation=90)
    # plt.yticks(np.arange(grid_frames),ytick_labels_1)
    cbar_fraction = 0.025
    cbar = plt.colorbar(fraction=cbar_fraction) 
    cbar.set_label('normalised frequency distribution')
    plt.title('{}\n fourier spectrum of signal binned over time\n'.format(filename))
    plt.ylabel('signal frequency, f / Hz')
    plt.xlabel('long time duration, t / s')
    plt.savefig('expt_ngrid_{}.pdf'.format(filetag),bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7,14))
    plt.imshow(path_grid, cmap='viridis');
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')  
    # plt.xticks(np.arange(long_timesteps),xtick_labels_1, rotation=90)
    # plt.yticks(np.arange(grid_frames),ytick_labels_1)
    plt.title('{}\n viterbi path through signal grid'.format(filename))
    plt.ylabel('signal frequency, f / Hz')
    plt.xlabel('long time duration, t / s')
    plt.savefig('expt_viterbi_path_{}.pdf'.format(filetag),bbox_inches='tight')
    plt.close()    
    
def tone_shift_check(filename, inj_tone, filetag=None):
    """checks for a frequency shift in an injected tone,
    filename must be a tone.csv recording"""
    if filetag is None:
        ft0 = filename.find('podo_')
        if ft0 == 0:
            filetag = filename[5:-4]
        else:
            filetag = filename[:-4]        

    mega = np.genfromtxt(filename,delimiter=',')
    times = mega[:,0]
    long_signal = mega[:,1]
    fps = len(mega)/mega[-1,0]

    # only take spectrum of first 5 seconds to save time
    fs0 = int(5*fps)
    raw_prob, raw_scale = fourier_spectrum_2(long_signal[:fs0], fps, return_spectrum=True)

    c0 = np.searchsorted(raw_scale, inj_tone-50)
    c1 = np.searchsorted(raw_scale, inj_tone+50)
    maxtone = raw_scale[c0+np.argmax(raw_prob[c0:c1])]

    fig, ax = plt.subplots(figsize=(14,7))
    ax.plot(raw_scale, raw_prob)
    ax.set(ylim=(0, 3),xlim=(inj_tone-50, inj_tone+50))
    ax.axvline(maxtone, color='red')
    plt.show()
    
def makeshift_comb(filename):
    """filters .csv signal with a makeshift comb filter,
    this does not work particularly well for speech"""
    mega = np.genfromtxt(filename,delimiter=',')
    times = mega[:,0]
    long_signal = mega[:,1]
    fps = len(mega)/mega[-1,0]

    f0_cut = 50
    # s.t. bottom is -3 dB, minimum desired response
    q_factor = 1

    # notch filter
    b, a = ssignal.iirnotch(f0_cut, q_factor, fps)
    filtered_signal = ssignal.filtfilt(b, a, long_signal)
    # freq_scale, freq_response = ssignal.freqz(b, a, fs=fps)
    # freq_dB = 20*np.log10(abs(freq_response))

    for n in range(2, 9):
        b, a = ssignal.iirnotch(n*f0_cut, q_factor, fps)
        filtered_signal = ssignal.filtfilt(b, a, filtered_signal)
        freq_scale, freq_response = ssignal.freqz(b, a, fs=fps)
        freq_dB = 20*np.log10(abs(freq_response))
        plt.plot(freq_scale, freq_dB)

    plt.xlim(0, 1000)
    plt.ylabel('amplitude / dB')
    plt.xlabel('frequency / Hz')
    plt.savefig('makeshift_comb.pdf')
    plt.close()

    wavfile.write('{}_makeshift_comb.wav'.format(filename), int(fps),
                  filtered_signal.astype('float32')/absmax(filtered_signal))
    
def wav_sanity_checks(filename='source_feynman(1).wav', filetag='feynman'):
    """sanity checks for basic wav reading and writing"""
    rate, signal = wavfile.read(filename)
    # mono np.nonzero(signal[:,0] - signal[:,1])
    signal = signal[:,0]

    mono_signal = signal[:,0]
    wavfile.write('{}_nochanges.wav'.format(filetag), rate, mono_signal)

    amplitude = np.iinfo(np.int16).max
    squeeze = signal/absmax(signal)*amplitude/2
    # scaled_data = amplitude * (squeeze-squeeze.mean())
    # shift = signal+signal.mean()/1000

    wavfile.write('{}_direct.wav'.format(filetag), rate, squeeze)

    # decrease rate from 44100 to around 16000
    audio_expt_rate = int(fps)
    # rate/audio_expt_rate ~= 2.7 
    rate_ratio = rate/audio_expt_rate
    # sample every rate_ratio values
    those = np.round(rate_ratio*np.arange(len(signal))).astype(int)
    those = those[those < len(signal)]
    sampled_signal = signal[those]

    wavfile.write('{}_sampled.wav'.format(filetag), audio_expt_rate, sampled_signal)
    
def wav_digitisation_check(infile_name='source_a440.wav', outfile_name=None):
    """checking that slower fps and digitisation
    doesn't significantly impact audio signal"""
    rate, signal = wavfile.read(infile_name)
    # mono np.nonzero(signal[:,0] - signal[:,1])
    if len(signal.shape) > 1:
        signal = signal[:,0]

    # decrease rate from 44100 to around 16000
    audio_expt_rate = int(rate)
    if rate > audio_expt_rate + 1000:
        # rate/audio_expt_rate ~= 2.7 
        rate_ratio = rate/audio_expt_rate
        print('rate_ratio:',rate_ratio)
        # sample every rate_ratio values
        those = np.round(rate_ratio*np.arange(len(signal))).astype(int)
        those = those[those < len(signal)]
        sampled_signal = signal[those]
    else:
        sampled_signal = signal
        audio_expt_rate = rate

    # sampled_signal_n = sampled_signal/absmax(sampled_signal)
    # sampled_signal_n -= sampled_signal_n.mean()

    # bin signal into channels
    intensity_channels = 100
    # centre = sampled_signal.mean()
    m0, m1 = sampled_signal.min(), sampled_signal.max()
    bins = np.linspace(m0-1e-10, m1+1e-10, intensity_channels)
    bin_lookup = np.digitize(sampled_signal, bins)
    digi_signal = bins[bin_lookup]/m1
    # digi_signal = digi_signal/intensity_channels
    # digi_signal = (digi_signal-0.5)*2*absmax(sampled_signal)+centre

    # data_to_audio = digi_signal

    # squeeze = data_to_audio/data_to_audio.max()
    # amplitude = np.iinfo(np.int16).max
    # scaled_data = amplitude * (squeeze-squeeze.mean())

    if outfile_name is None:
        outfile_name = '{}_digi.wav'.format(infile_name[:-4])
    wavfile.write(outfile_name, audio_expt_rate, digi_signal)

    fig, (ax0, ax1) = plt.subplots(2,figsize=(14,14))
    # a,b,b1 = 0, int(len(signal)/100), int(len(signal)/40+10000)
    a, b = 0, int(len(signal)/50)
    ax0.plot(signal[a:b])
    ax1.plot(digi_signal[a:b])
    plt.savefig('{}_digitised_simulation.pdf'.format(infile_name[:-4]))
    plt.show()
    plt.close(fig)
    
def psd_plot(filename):
    """find power spectral density (psd) of noise
    requires noise recording, like podo_14_6.csv"""
    mega = np.genfromtxt(filename,delimiter=',')
    times = mega[:,0]
    long_signal = mega[:,1]
    fps = len(mega)/mega[-1,0]

    c0, c1 = 10000, 400000
    cut = long_signal[c0:c1]
    cut_time = times[c0:c1]-times[c0]

    fig, ax = plt.subplots(figsize=(14,7))
    ax.psd(cut, int(2**13), fps)
    ax.set(ylabel='power spectral density / dB/Hz',
           xlabel='frequency / Hz')
    ax.set_xlim(0, 2000)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.xaxis.label.set_size(26)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.yaxis.label.set_size(26) 
    plt.savefig(filename[:-4]+'.pdf', bbox_inches='tight')
    plt.close(fig)
    
def butter_filter_recording(filename, filetag=None, produce_plots=False,
                           cut_off_f1=3000):
    """butterworth filters time series from .csv and saves .wav recordings"""
    if filetag is None:
        ft0 = filename.find('podo_')
        if ft0 == 0:
            filetag = filename[5:-4]
        else:
            filetag = filename[:-4]        

    mega = np.genfromtxt(filename,delimiter=',')
    times = mega[:,0]
    long_signal = mega[:,1]
    fps = len(mega)/mega[-1,0]
    
    total_frames = len(long_signal)
    duration = total_frames/fps
    #print('duration: {:.1f}s, fps: {:.1f}/s, frames: {}'.format(duration, fps, total_frames))

    # create butterworth bandpass filter
    
    butter_order = 5
    # mains noise at 50, 100Hz
    cut_off_f0 = 150
    # old phone lines at 3kHz
    cut_off_f1 = 3000
    filter_coeff_b, filter_coeff_a = ssignal.butter(butter_order,
                                                    (cut_off_f0, cut_off_f1),
                                                    btype='bandpass',
                                                    fs=fps)

    # could also use simple lfilter which will cause a phase change
    filtered_signal = ssignal.filtfilt(filter_coeff_b, filter_coeff_a, long_signal)

    wavfile.write('podo_{}_raw.wav'.format(filetag), int(fps),
                  long_signal.astype('float32')/absmax(long_signal))
    wavfile.write('podo_{}_filtered.wav'.format(filetag), int(fps),
                  filtered_signal.astype('float32')/absmax(filtered_signal))
    
    if produce_plots:
        w, h = ssignal.freqz(filter_coeff_b, filter_coeff_a, fs=fps)

        # plt.plot(w, h)
        # plt.semilogx(w, np.log(abs(h)))
        fig, ax = plt.subplots()
        # 10*log10(power), power = amp^2, therefore: 20*log10(amp)
        ax.semilogx(w, 20*np.log10(abs(h)))
        ax.grid(which='both')
        ax.axvline(cut_off_f0, color='green')
        ax.axvline(cut_off_f1, color='green')
        ax.set(title='butterworth bandpass (150Hz-3kHz) filter response',
               ylabel = 'amplitude / dB',
               xlabel = 'frequency, Hz')
        fig.savefig('butterworth_{}_{}.pdf'.format(cut_off_f0, cut_off_f1))
        plt.close(fig)

        fig, (ax0, ax1) = plt.subplots(2, figsize=(14,7), sharex=True)
        ax0.plot(times, long_signal)
        ax1.plot(times, filtered_signal)
        ax0.set_title('{} time series, before and after filter'.format(filetag))
        ax1.set_xlabel('time, t / s')
        ax0.set_xlim(0, 2)
        ax1.set_xlim(0, 2)
        fig.savefig('filter_timeseries_{}.pdf'.format(filetag))
        plt.close(fig)

        # only take spectrum of first 5 seconds to save time
        fs0 = int(5*fps)
        raw_prob, raw_scale = fourier_spectrum_2(long_signal[:fs0], fps, return_spectrum=True)
        filtered_prob, filtered_scale = fourier_spectrum_2(filtered_signal[:fs0], fps, return_spectrum=True)

        fig, (ax0, ax1) = plt.subplots(2, figsize=(14,7), sharex=True)
        ax0.plot(raw_scale, raw_prob)
        ax1.plot(filtered_scale, filtered_prob)
        ax0.set(title='{} spectrum, before and after filter'.format(filetag),
                xlim=(0, 1000), ylim=(0, 0.8))
        ax1.set(xlabel='frequency, f / Hz', ylim=(0, 0.8))
        fig.savefig('filter_spectrum_{}.pdf'.format(filetag))  
        plt.close(fig)
        
def butter_filter_plot(butter_order=5, cut_off_f0=150, cut_off_f1=3000, fps=16000):
    """create butterworth bandpass filter and plot response"""    
    filter_coeff_b, filter_coeff_a = ssignal.butter(butter_order,
                                                    (cut_off_f0, cut_off_f1),
                                                    btype='bandpass',
                                                    fs=fps)
    w, h = ssignal.freqz(filter_coeff_b, filter_coeff_a, fs=fps)

    # plt.plot(w, h)
    # plt.semilogx(w, np.log(abs(h)))
    fig, ax = plt.subplots()
    # 10*log10(power), power = amp^2, therefore: 20*log10(amp)
    ax.semilogx(w, 20*np.log10(abs(h)))
    ax.grid(which='both')
    ax.axvline(cut_off_f0, color='green')
    ax.axvline(cut_off_f1, color='green')
    ax.set(ylabel = 'amplitude / dB',
           xlabel = 'frequency / Hz')
    ax.xaxis.set_tick_params(labelsize=14)
    ax.xaxis.label.set_size(16)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.yaxis.label.set_size(16)    
    fig.savefig('butterworth_{}_{}.pdf'.format(cut_off_f0, cut_off_f1), bbox_inches='tight')
    plt.close(fig)
    
def logmmse_filter(filename='aa_melatos.csv', filetag=None):
    """applies logMMSE speech enhancement filter from existing implementation"""
    if filetag is None:
        ft0 = filename.find('podo_')
        if ft0 == 0:
            filetag = filename[5:-4]
        else:
            filetag = filename[:-4]   

    mega = np.genfromtxt(filename,delimiter=',')
    times = mega[:,0]
    long_signal = mega[:,1]
    fps = len(mega)/mega[-1,0]

    wavfile.write('podo_{}_raw.wav'.format(filetag), int(fps),
                  long_signal.astype('float32')/absmax(long_signal))

    logmmse_signal = logmmse(long_signal.astype('float32'), int(fps))

    wavfile.write('podo_{}_logmmse.wav'.format(filetag), int(fps),
                  logmmse_signal.astype('float32')/absmax(logmmse_signal))

    fig, (ax0, ax1) = plt.subplots(2, figsize=(14,7), sharex=True)
    ax0.plot(times, long_signal)
    ax1.plot(np.arange(len(logmmse_signal))/fps, logmmse_signal)
    # ax0.set_title('{} time series, before and after logMMSE filter'.format(filetag))
    ax1.set_xlabel('time, t / s')
    ax0.set_xlim(0, 1.1)
    ax1.set_xlim(0, 1.1)
    fig.subplots_adjust(hspace=0.1)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.xaxis.label.set_size(22)
    ax0.set_ylabel('voltage signal\n from ADC')
    ax0.yaxis.label.set_size(22)    
    ax0.yaxis.set_tick_params(labelsize=20)
    ax1.set_ylabel('digital intensity')
    ax1.yaxis.label.set_size(22)        
    ax1.yaxis.set_tick_params(labelsize=20)
    fig.savefig('filter_timeseries_{}.pdf'.format(filetag), bbox_inches='tight')
    plt.close(fig)

    fs0 = int(5*fps)
    raw_prob, raw_scale = fourier_spectrum_2(long_signal[:fs0], fps, return_spectrum=True)
    filtered_prob, filtered_scale = fourier_spectrum_2(logmmse_signal[:fs0], fps, return_spectrum=True)

    fig, (ax0, ax1) = plt.subplots(2, figsize=(14,7), sharex=True)
    ax0.plot(raw_scale, raw_prob)
    ax1.plot(filtered_scale, filtered_prob)
    ax0.set(xlim=(0, 2100), ylim=(0, 2))
    ax1.set(xlabel='frequency, f / Hz', ylim=(0, 2))
    fig.subplots_adjust(hspace=0.15)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.xaxis.label.set_size(22)
    ax0.yaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    fig.text(0.06, 0.5, 'Fourier amplitude', va='center', rotation='vertical', fontsize=22)
    fig.savefig('filter_spectrum_{}.pdf'.format(filetag), bbox_inches='tight')  
    plt.close(fig)
    
def check_filters_on_source(filename='source_melatos.wav', filetag='source_melatos'):
    """run the filter process over a source file to compare to the experimental recording"""
    fps, long_signal = wavfile.read(filename)

    logmmse_signal = logmmse(long_signal.astype('float32'), int(fps))

    og_signal = long_signal[:len(logmmse_signal)][:,0]
    long_signal = logmmse_signal[:,0]
    times = np.arange(len(long_signal))/fps

    butter_order = 5
    # mains noise at 50, 100Hz
    cut_off_f0 = 100
    # old phone lines at 3kHz
    cut_off_f1 = 1000 
    filter_coeff_b, filter_coeff_a = ssignal.butter(butter_order,
                                                    (cut_off_f0, cut_off_f1),
                                                    btype='bandpass',
                                                    fs=fps)

    # could also use the simple lfilter() which will cause a phase change
    # filt filt applies signal then its adjoint (conjugate transpose)
    filtered_signal = ssignal.filtfilt(filter_coeff_b, filter_coeff_a, long_signal)

    wavfile.write('podo_{}_logmmse.wav'.format(filetag), int(fps),
                  long_signal.astype('float32')/absmax(long_signal))

    wavfile.write('podo_{}_logmmse_butter.wav'.format(filetag), int(fps),
                  filtered_signal.astype('float32')/absmax(filtered_signal))

    fig, (ax0, ax1) = plt.subplots(2, figsize=(14,7), sharex=True)
    ax0.plot(times, long_signal)
    ax1.plot(times, filtered_signal)
    ax0.set_title('{} time series, before and after filter'.format(filetag))
    ax1.set_xlabel('time, t / s')
    ax0.set_xlim(0, 2)
    ax1.set_xlim(0, 2)
    fig.savefig('filter_timeseries_{}.pdf'.format(filetag))
    plt.close(fig)

    # only take spectrum of first 5 seconds to save time
    fs0 = int(5*fps)
    raw_prob, raw_scale = fourier_spectrum_2(long_signal[:fs0], fps, return_spectrum=True)
    filtered_prob, filtered_scale = fourier_spectrum_2(filtered_signal[:fs0], fps, return_spectrum=True)

    fig, (ax0, ax1) = plt.subplots(2, figsize=(14,7), sharex=True)
    ax0.plot(raw_scale, raw_prob/absmax(raw_prob))
    ax1.plot(filtered_scale, filtered_prob/absmax(filtered_prob))
    ax0.set(title='{} spectrum, before and after filter'.format(filetag),
            xlim=(0, 2000), ylim=(0, 1))
    ax1.set(xlabel='frequency, f / Hz', ylim=(0, 1))
    fig.savefig('filter_spectrum_{}.pdf'.format(filetag))  
    plt.close(fig)

if __name__ == "__main__":
    psd_plot('podo_14_6.csv')
    butter_filter_plot()
    logmmse_filter('aa_melatos.csv')
    logmmse_filter('aa_jam_track.csv')
