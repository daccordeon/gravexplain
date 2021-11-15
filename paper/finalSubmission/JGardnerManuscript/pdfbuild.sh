#!/bin/bash

##########################
# build the pdf by doing #
#     bash pdfbuild.sh   #
##########################

pdflatex JGardner.tex
bibtex JGardner
pdflatex JGardner.tex
pdflatex JGardner.tex
