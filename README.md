# OTProf: estimating high-resolution profiles of optical turbulence ($C_n^2$) from reanalysis using deep learning

Code accompanying the publication

> Pierzyna, Maximilian, et al. “OTProf: Estimating High-Resolution Profiles of Optical Turbulence ($C_n^2$) from Reanalysis Using Deep Learning.” arXiv:2604.09346, arXiv, 10 Apr. 2026. arXiv.org, https://doi.org/10.48550/arXiv.2604.09346.

This repository covers two main parts of the project:

1. **Dataset generation** using the Weather Research and Forecasting model (WRF) using the [`wrf-massive`](https://github.com/mpierzyna/wrf-massive) package is detailed in [`data/nl/wrf`](data/nl/wrf).
2. **Deep learning pipeline** is addressed in [`workspaces/nl`](workspaces/nl) using the `otprof` package. TODO