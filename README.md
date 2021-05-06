# GravExplain
## README.md
*GravExplain: Continuous gravitational wave searches in a table-top experiment*

James Gardner: OzGrav-Australian National University

Hannah Middleton, Changrong Liu, Andrew Melatos, Robin Evans, William Moran: OzGrav-Melbourne University and Department of Electrical and Electronic Engineering, University of Melbourne
*(these authors all contributed to the code, for the full list of authors and affiliations see the paper)*

April 2021

Current build found [here](https://github.com/daccordeon/gravexplain).

---
Guide to replicating results, please follow exactly:

- Download the latest source code from the repository. Download the sample files (three .csv, two .mp4, one .wav) as sample3.zip available [here](https://github.com/daccordeon/gravexplain/releases/download/v1.3/sample3.zip), and extract them into the same directory as the source code.
- Execute main.py with python3, this can typically be done via the command line with "python3 main.py" when in the directory containing all of the source code and sample files.
- The newest version of the code in found in the Jupyter Notebooks, to generate these plots run all cells in viterbi_video_analysis.ipynb and optical_microphone_filters.ipynb
- Execute noiseclear.m with MATLAB
- This should produce all of the figures, as .pdf files, of the experimental results used as well as .wav recordings of the optical microphone results.
- This ends the guide to replicating results directly. The user is invited to use the functionality of the various scripts to investigate their own recordings and, hopefully, reproduce the results shown.
- To make your own recording with the photodiode and optical microphone set-up as described in the paper. Execute raspberry_pi_photodiode_adc.py with python3 inside the Raspberry Pi. This will produce a .csv of the recorded signal which can be transferred over for analysis.
- Contact the authors for any technical enquiries at <u6069809@anu.edu.au>.

Requirements:
- ipython==5.5.0
- jupyter==1.0.0
- jupyter-client==5.2.2
- jupyter-console==6.0.0
- jupyter-core==4.4.0
- logmmse==1.4 (from [here](https://github.com/wilsonchingg/logmmse))
- matplotlib==3.0.3
- numpy==1.16.2
- scipy==1.2.1
- tqdm==4.33.0
- matlab==R2020a

---
Directory structure:
```bash
.
├── .gitignore
├── README.md
├── LICENSE
├── paper
│   ├── figures
│   │   └── ...
│   ├── ifo-appendix.tex
│   ├── ifo-complexAudio.tex
│   ├── ifo-conclusions.tex
│   ├── ifoDemoBib.bib
│   ├── ifo-futureWork.tex
│   ├── ifo-introduction.tex
│   ├── ifo-singleTone.tex
│   ├── ifo-supplementary-material.tex
│   ├── ifo-tableTopGWs.tex
│   ├── ifo-wanderingTone.tex
│   ├── myunsrt.bst
│   ├── paper_mainNotes.bib
│   ├── paper_main.tex
│   └── response
│       └── AJPRefereeResponse.tex
└── source
    ├── main.py
    ├── viterbi_video_analysis.ipynb
    ├── viterbi_video_analysis.py (applies the Viterbi algorithm to an .mp4 from the webcam)
    ├── raspberry_pi_photodiode_adc.py (reads photodiode signal through ADC by SPI pins, saves to .csv)
    ├── optical_microphone_filters.ipynb
    ├── optical_microphone_filters.py (filters signal from optical microphone produced by raspberry_pi_photodiode_adc.py, also applies the Viterbi algorithm)
    ├── noiseclear.m (advanced filtering of optical microphone signal)
    ├── wienerFilt.m (performs Wiener filtering in noiseclear.m)
    ├── mock_continuous_signal.ipynb
    ├── mock_continuous_signal.py (simulates applying the Viterbi algorithm to a noisy signal)
    └── defunct_open_cv.ipynb (defunct functions produced while learning the openCV library)
```
[//]: # (tree -I '*.pdf|*.png|*.svg|*.jpg')
