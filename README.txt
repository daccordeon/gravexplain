GravExplain: explaining gravitational wave data science to a non-specialist audience
James Gardner, ANU
Andrew Melatos and Hannah Middleton, OzGrav @ Melbourne Uni
Summer 2019/2020
README

Current build found at:
https://github.com/daccordeon/gravexplain

Directory structure:
gravexplain/
    source/
        mock_continuous_signal.py
    plots/
        (copies of all plots saved by scripts
         from expt_4 onwards time series are recorded from a point, not tracked)
    
    mock_continuous_signal.ipynb
    tracker_time_series.ipynb
    open_cv.ipynb
    experiment_viterbi.ipynb

    reading_notes.txt
    hardware.txt

    .gitignore
    LICENSE
    README.txt

- - -
Guide to replicating results, please follow exactly

GravExplain uses python 3.6.8 in jupyter notebook and has the requirements:
ipython==5.5.0
jupyter==1.0.0
jupyter-client==5.2.2
jupyter-console==6.0.0
jupyter-core==4.4.0
matplotlib==3.0.3
numpy==1.16.2
pandas==0.24.2
tqdm==4.33.0

