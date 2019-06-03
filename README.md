# KindAP
This repository includes the source code (Python and MATLAB) for solving a new clustering model k-indicators, with a concentration on KindAP algorithms (K-indicators by Alternating Projections).

## MATLAB version
K-indicators are written in functions, but users need to create data directories under $KIND_PATH/matlab_data/. All files are saved as .mat type with one data matrix and one ground truth label.
Run the demos and tests.

### Dependencies
After MATLAB R2017b, otherwise, some functions may not be available. 

## Python version  
Install the package Kind
```
pip install -e Kind
```
for an editable version.

The use of K-indicators clustering is similar as using sklearn clustering packages, and more detailed documentations are on-going.

### Dependencies
sklearn, numpy, six, munkres, scipy, see `setup.py`.

## Copyright:
Yuchen Yang, Feiyu Chen, Yin Zhang
