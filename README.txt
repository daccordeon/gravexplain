GravExplain: explaining gravitational wave data science to a non-specialist audience
James Gardner, ANU
Hannah Middleton and Andrew Melatos, OzGrav @ Melbourne Uni
Summer 2020
README

Current build found at:
https://github.com/daccordeon/gravexplain

Directory structure:
gravexplain/
    source/
        [py executable script versions of all .ipynb notebooks]
        raspberry_pi_photodiode_adc.py
            (reads photodiode signal through ADC by SPI pins, saves to .csv)
    paper/
        paper_main.tex
        ifoDemoBib.bib
        myunsrt.bst
    
    mock_continuous_signal.ipynb
        (simulates applying the Viterbi algorithm to a noisy signal)
    tracker_time_series.ipynb
        (defunct; saves signal and spectrum plots from a .csv file)
    open_cv.ipynb
        (defunct; various functions produced while learning the openCV library)
    experiment_viterbi.ipynb
        (applies the Viterbi algorithm to an .mp4 from the webcam method)
    experiment_photodiode.ipynb
        (filters signal from optical microphone produced by raspberry_pi_photodiode_adc.py,
        also applies the Viterbi algorithm)

    .gitignore
    LICENSE
    README.txt

- - -
Guide to replicating results, please follow exactly

- - -
GravExplain uses python 3.6.8 in jupyter notebook and has the requirements:
ipython==5.5.0
jupyter==1.0.0
jupyter-client==5.2.2
jupyter-console==6.0.0
jupyter-core==4.4.0
logmmse==1.4
matplotlib==3.0.3
numpy==1.16.2
scipy==1.2.1
tqdm==4.33.0
