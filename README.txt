GravExplain: Continuous gravitational wave searches in a table-top experiment
James Gardner: PhB Science - Australian National University
Hannah Middleton, Andrew Melatos, Robin Evans, Bill Moran: OzGrav-Melbourne
February 2020
README

Current build found at:
https://github.com/daccordeon/gravexplain

Directory structure:
gravexplain/
    source/
        main.py
            (produces the experimental result figures from scripts given experimental data)

        raspberry_pi_photodiode_adc.py
            (reads photodiode signal through ADC by SPI pins, saves to .csv)

        viterbi_video_analysis.py
            (applies the Viterbi algorithm to an .mp4 from the webcam method)
        optical_microphone_filters.py
            (filters signal from optical microphone produced by raspberry_pi_photodiode_adc.py, also applies the Viterbi algorithm)
        mock_continuous_signal.py
            (simulates applying the Viterbi algorithm to a noisy signal)

        [also contains .ipynb versions of above three .py scripts]

        defunct_open_cv.ipynb
            (various functions produced while learning the openCV library)
    
    paper/
        figures/
            [all .pdf figures used in paper_main.tex and some corresponding .svg]

        paper_main.tex
        ifoDemoBib.bib
        myunsrt.bst

    .gitignore
    LICENSE
    README.txt

- - -
Guide to replicating results, please follow exactly.

Download the latest source code from the repository. Download the sample files (two .csv and two .mp4) as sample.zip from the release below, and extract them into the same directory as the source code.
https://github.com/daccordeon/gravexplain/releases/download/v1.0/sample.zip

Execute main.py with python3, this can typically be done via the command line with "python3 main.py" when in the directory containing all of the source code and sample files.

This should produce all of the figures, as .pdf files, of the experimental results used as well as .wav recordings of the optical microphone results.

This ends the guide to replicating results directly. The user is invited to use the functionality of the various scripts to investigate their own recordings and, hopefully, reproduce the results shown.

To make your own recording with the photodiode and optical microphone set-up as described in the paper. Execute raspberry_pi_photodiode_adc.py with python3 inside the Raspberry Pi. This will produce a .csv of the recorded signal which can be transferred over for analysis.

- - -
GravExplain uses python 3.6.8 in jupyter notebook and has the requirements:
ipython==5.5.0
jupyter==1.0.0
jupyter-client==5.2.2
jupyter-console==6.0.0
jupyter-core==4.4.0
logmmse==1.4
    (implementation from https://github.com/wilsonchingg/logmmse)
matplotlib==3.0.3
numpy==1.16.2
scipy==1.2.1
tqdm==4.33.0
