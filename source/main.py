#!/usr/bin/env python3
"""main.py
James Gardner 2020
ANU / Melbourne Uni

performs entire analysis pipeline if executed
i.e. run the following command in terminal:
python3 main.py
with the following files present in cwd:
expt_viterbi_test_webcam.mp4 expt_4_0209.mp4 aa_melatos.csv aa_jam_track.csv
this will produce the respective plots used in the GravExplain paper
"""

import os.path
import viterbi_video_analysis
import optical_microphone_filters

if __name__ == "__main__":
    
    if (os.path.isfile('expt_4_0209.mp4') and os.path.isfile('expt_viterbi_test_webcam.mp4')
        and os.path.isfile('aa_melatos.csv') and os.path.isfile('aa_jam_track.csv')):
        print('Welcome to GravExplain main.py, all sample files are present, beginning analysis now')
        
        viterbi_video_analysis.webcam_video_analysis('expt_4_0209.mp4', point=(230,240))
        viterbi_video_analysis.PointViterbi('expt_viterbi_test_webcam.mp4', scanning_range=1).plot()

        optical_microphone_filters.psd_plot()
        optical_microphone_filters.butter_filter_plot()
        optical_microphone_filters.logmmse_filter('aa_melatos.csv')
        optical_microphone_filters.logmmse_filter('aa_jam_track.csv')           
        
    else:
        print ('Welcome to GravExplain main.py, some or all of the sample files are missing, please consult README.txt')     
