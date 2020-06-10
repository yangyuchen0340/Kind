# Introduction
This repository includes the source code (Python and MATLAB) for cluster analysis based on k-indicators, with a concentration on KindAP algorithms (K-indicators by Alternating Projections). The first publication of this study is on https://arxiv.org/pdf/1906.00938.pdf .

# MATLAB version

## Installation
Download and save the repository to $KIND_PATH. Run `compilepath.m` to add path to MATLAB.

## Dependencies
Versions after MATLAB R2017b, otherwise, some functions may not be available. 


# Python version  
## Installation
Download and install the package Kind at $KIND_PATH.
```
pip install -e Kind
```
for an editable version.

The API design of Kind is similar as scikit-learn for cluster analysis and more detailed documentations are on-going.

## Dependencies
sklearn, numpy, six, munkres, scipy, see `setup.py` for details.

## Tests
Users can create data directories called UCI under $KIND_PATH/matlab_data/. All files are saved as .mat type with one data matrix named `fea` and one ground truth label named `idxg`. 
Run the demos and tests. For `demo.py`, users could modify the address of `datadir` variable according to the data directories you have created. 
Tests that do not rely on MATLAB and real data are still ongoing. For m files tests, users could directly run the scripts on MATLAB.


# Copyright:
Yuchen Yang (author), Feiyu Chen, Yin Zhang
